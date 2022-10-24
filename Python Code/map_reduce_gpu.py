# Copyright 2019-2022 ETH Zurich and the DaCe authors. All rights reserved.
""" Contains classes and functions that implement the map-reduce-fusion transformation
    utilizing GPU warp primitives. 
"""

import copy
from dace import data, dtypes, nodes
from dace.sdfg import SDFG, SDFGState
from dace.memlet import Memlet
from dace.sdfg import nodes
from dace.properties import Property, make_properties
from dace.sdfg import SDFG
from dace.sdfg import utils as sdutil
from dace.symbolic import symstr
from dace.transformation import transformation as tr
from dace.transformation.passes import pattern_matching as pm

from dace.transformation.dataflow.map_collapse import MapCollapse
from dace.transformation.dataflow.map_fusion import MapFusion


@make_properties
class MapReduceFusionGPU(tr.SingleStateTransformation):
    """ Implements the map-reduce-fusion transformation utilizing GPU warp primitives.
        Fuses a map with an immediately following reduction, where the array
        between the map and the reduction is not used anywhere else.
    """

    no_init = Property(dtype=bool,
                       default=False,
                       desc='If enabled, does not create initialization states '
                       'for reduce nodes with identity')

    tasklet = tr.PatternNode(nodes.Tasklet)
    tmap_exit = tr.PatternNode(nodes.MapExit)
    in_array = tr.PatternNode(nodes.AccessNode)

    import dace.libraries.standard as stdlib  # Avoid import loop
    reduce = tr.PatternNode(stdlib.Reduce)

    out_array = tr.PatternNode(nodes.AccessNode)

    @classmethod
    def expressions(cls):
        return [sdutil.node_path_graph(cls.tasklet, cls.tmap_exit, cls.in_array, cls.reduce, cls.out_array)]

    def can_be_applied(self, graph, expr_index, sdfg, permissive=False):
        tmap_exit = self.tmap_exit
        in_array = self.in_array
        reduce_node = self.reduce
        tasklet = self.tasklet
        out_array = self.out_array

        # Make sure that the array is only accessed by the map and the reduce
        if any([src != tmap_exit for src, _, _, _, memlet in graph.in_edges(in_array)]):
            return False
        if any([dest != reduce_node for _, _, dest, _, memlet in graph.out_edges(in_array)]):
            return False

        tmem = next(e for e in graph.edges_between(tasklet, tmap_exit) if e.data.data == in_array.data).data

        # Make sure that the transient is not accessed anywhere else
        # in this state or other states
        if not permissive and (
                len([n for n in graph.nodes() if isinstance(n, nodes.AccessNode) and n.data == in_array.data]) > 1
                or in_array.data in sdfg.shared_transients()):
            return False

        # If memlet already has WCR and it is different from reduce node,
        # do not match
        if tmem.wcr is not None and tmem.wcr != reduce_node.wcr:
            return False

        # Verify that reduction ranges match tasklet map
        tout_memlet = graph.in_edges(in_array)[0].data
        rin_memlet = graph.out_edges(in_array)[0].data
        if tout_memlet.subset != rin_memlet.subset:
            return False

        # NOTE: We constrain the transformation to match 1-D Map with N-D -> (N-1)-D reduction.
        if len(tmap_exit.map.params) > 1:
            return False
        out_desc = sdfg.arrays[out_array.data]
        len_output = 0 if isinstance(out_desc, data.Scalar) else len(out_desc.shape)
        if len(sdfg.arrays[in_array.data].shape) - len_output != 1:
            return False

        return True

    def match_to_str(self, graph):
        return ' -> '.join(str(node) for node in [self.tasklet, self.tmap_exit, self.reduce])

    def apply(self, graph: SDFGState, sdfg: SDFG):
        tmap_exit = self.tmap_exit
        in_array = self.in_array
        reduce_node = self.reduce
        out_array = self.out_array

        # Add CUDA-related symbols and constants
        for s in ('BlockDim', 'GridDim'):
            try:
                sdfg.add_symbol(s, dtypes.int32)
            except FileExistsError:
                pass
        try:
            sdfg.add_constant('WarpSize', 32)
        except FileExistsError:
            pass

        identity = reduce_node.identity or 0
        reduce_node.implementation = 'CUDA(shuffle)'

        # GPUTransform subgraph
        from dace.transformation.dataflow import GPUTransformMap
        from dace.transformation.interstate import InlineSDFG
        tmap_entry = graph.entry_node(tmap_exit)
        sdfg.arrays[in_array.data].storage = dtypes.StorageType.GPU_Global
        out_desc = sdfg.arrays[out_array.data]
        if out_desc.storage not in (dtypes.StorageType.GPU_Global, dtypes.StorageType.CPU_Pinned):
            copy_desc = copy.deepcopy(out_desc)
            name = sdfg.add_datadesc(out_array.data, copy_desc, find_new_name=True)
            access = graph.add_access(name)
            for e in graph.out_edges(out_array):
                if e.data.data == out_array.data:
                    e.data.data = name
                graph.add_edge(access, e.src_conn, e.dst, e.dst_conn, e.data)
                graph.remove_edge(e)
            graph.add_nedge(out_array, access, Memlet.from_array(name, copy_desc))
            sdfg.arrays[out_array.data].storage = dtypes.StorageType.GPU_Global
        if tmap_entry.map.schedule != dtypes.ScheduleType.GPU_Device:
            GPUTransformMap.apply_to(sdfg, map_entry=tmap_entry)
        reduce_node.expand(sdfg, graph)
        nsdfg = graph.out_edges(in_array)[0].dst
        if isinstance(nsdfg, nodes.NestedSDFG):
            nsdfg.sdfg.simplify()
            InlineSDFG.apply_to(sdfg, nested_sdfg=nsdfg)

        # Get reductions Maps
        reduction_entries = []
        for i, e in enumerate(graph.memlet_path(graph.out_edges(in_array)[0])):
            reduction_entries.append(e.dst)
            if i == 2:
                break
        rientry, rjentry, rtentry = reduction_entries

        # Tile Map
        from GPU_tiling_transformation import GPU_Tiling
        mt_entry, mi_entry, mj_entry = GPU_Tiling.apply_to(sdfg, map_entry=tmap_entry)


        MapFusion.apply_to(sdfg, first_map_exit=graph.exit_node(mi_entry), array=in_array, second_map_entry=rientry)
        access = graph.in_edges(rjentry)[0].src
        MapFusion.apply_to(sdfg, first_map_exit=graph.exit_node(mj_entry), array=access, second_map_entry=rjentry)
        access = graph.in_edges(rtentry)[0].src
        MapFusion.apply_to(sdfg, first_map_exit=graph.exit_node(mt_entry), array=access, second_map_entry=rtentry)
        
        # NOTE: Hacky, need to investigate why Map params become symbols in MapFusion
        for s in mi_entry.map.params + mj_entry.map.params:
            del sdfg.symbols[s]
        
        # # Set nodes to remove according to the expression index
        # nodes_to_remove = [in_array]
        # nodes_to_remove.append(reduce_node)

        # memlet_edge = None
        # for edge in graph.in_edges(tmap_exit):
        #     if edge.data.data == in_array.data:
        #         memlet_edge = edge
        #         break
        # if memlet_edge is None:
        #     raise RuntimeError('Reduction memlet cannot be None')

        # # Find which indices should be removed from new memlet
        # input_edge = graph.in_edges(reduce_node)[0]
        # axes = reduce_node.axes or list(range(len(input_edge.data.subset)))
        # array_edge = graph.out_edges(reduce_node)[0]

        # # Delete relevant edges and nodes
        # graph.remove_nodes_from(nodes_to_remove)

        # # Delete relevant data descriptors
        # for node in set(nodes_to_remove):
        #     if isinstance(node, nodes.AccessNode):
        #         # try to delete it
        #         try:
        #             sdfg.remove_data(node.data)
        #         # will raise ValueError if the datadesc is used somewhere else
        #         except ValueError:
        #             pass

        # # Filter out reduced dimensions from subset
        # filtered_subset = [dim for i, dim in enumerate(memlet_edge.data.subset) if i not in axes]
        # if len(filtered_subset) == 0:  # Output is a scalar
        #     filtered_subset = [(0, 0, 1)]

        # # Modify edge from tasklet to map exit
        # memlet_edge.data.data = out_array.data
        # memlet_edge.data.wcr = reduce_node.wcr
        # memlet_edge.data.subset = type(memlet_edge.data.subset)(filtered_subset)

        # # Add edge from map exit to output array
        # graph.add_edge(
        #     memlet_edge.dst, 'OUT_' + memlet_edge.dst_conn[3:], array_edge.dst, array_edge.dst_conn,
        #     Memlet.simple(array_edge.data.data,
        #                   array_edge.data.subset,
        #                   num_accesses=array_edge.data.num_accesses,
        #                   wcr_str=reduce_node.wcr))

        # # Add initialization state as necessary
        # if not self.no_init and reduce_node.identity is not None:
        #     init_state = sdfg.add_state_before(graph)
        #     init_state.add_mapped_tasklet(
        #         'freduce_init',
        #         [('o%d' % i, '%s:%s:%s' % (r[0], r[1] + 1, r[2])) for i, r in enumerate(array_edge.data.subset)], {},
        #         '__out = %s' % reduce_node.identity, {
        #             '__out':
        #             Memlet.simple(array_edge.data.data, ','.join(
        #                 ['o%d' % i for i in range(len(array_edge.data.subset))]))
        #         },
        #         external_edges=True)


# class MapWCRFusion(pm.SingleStateTransformation):
#     """ Implements the map expanded-reduce fusion transformation.
#         Fuses a map with an immediately following reduction, where the array
#         between the map and the reduction is not used anywhere else, and the
#         reduction is divided to two maps with a WCR, denoting partial reduction.
#     """

#     tasklet = pm.PatternNode(nodes.Tasklet)
#     tmap_exit = pm.PatternNode(nodes.MapExit)
#     in_array = pm.PatternNode(nodes.AccessNode)
#     rmap_in_entry = pm.PatternNode(nodes.MapEntry)
#     rmap_in_tasklet = pm.PatternNode(nodes.Tasklet)
#     rmap_in_cr = pm.PatternNode(nodes.MapExit)
#     rmap_out_entry = pm.PatternNode(nodes.MapEntry)
#     rmap_out_exit = pm.PatternNode(nodes.MapExit)
#     out_array = pm.PatternNode(nodes.AccessNode)

#     @classmethod
#     def expressions(cls):
#         return [
#             # Map, then partial reduction of axes
#             sdutil.node_path_graph(cls.tasklet, cls.tmap_exit, cls.in_array, cls.rmap_out_entry, cls.rmap_in_entry,
#                                    cls.rmap_in_tasklet, cls.rmap_in_cr, cls.rmap_out_exit, cls.out_array)
#         ]

#     def can_be_applied(self, graph, expr_index, sdfg, permissive=False):
#         tmap_exit = self.tmap_exit
#         in_array = self.in_array
#         rmap_entry = self.rmap_out_entry

#         # Make sure that the array is only accessed by the map and the reduce
#         if any([src != tmap_exit for src, _, _, _, memlet in graph.in_edges(in_array)]):
#             return False
#         if any([dest != rmap_entry for _, _, dest, _, memlet in graph.out_edges(in_array)]):
#             return False

#         # Make sure that there is a reduction in the second map
#         rmap_cr = self.rmap_in_cr
#         reduce_edge = graph.in_edges(rmap_cr)[0]
#         if reduce_edge.data.wcr is None:
#             return False

#         # Make sure that the transient is not accessed anywhere else
#         # in this state or other states
#         if not permissive and (
#                 len([n for n in graph.nodes() if isinstance(n, nodes.AccessNode) and n.data == in_array.data]) > 1
#                 or in_array.data in sdfg.shared_transients()):
#             return False

#         # Verify that reduction ranges match tasklet map
#         tout_memlet = graph.in_edges(in_array)[0].data
#         rin_memlet = graph.out_edges(in_array)[0].data
#         if tout_memlet.subset != rin_memlet.subset:
#             return False

#         return True

#     def match_to_str(self, graph):
#         return ' -> '.join(str(node) for node in [self.tasklet, self.tmap_exit, self.rmap_in_cr])

#     def apply(self, graph: SDFGState, sdfg: SDFG):
#         # To apply, collapse the second map and then fuse the two resulting maps
#         map_collapse = MapCollapse()
#         map_collapse.setup_match(
#             sdfg, self.sdfg_id, self.state_id, {
#                 MapCollapse.outer_map_entry: graph.node_id(self.rmap_out_entry),
#                 MapCollapse.inner_map_entry: graph.node_id(self.rmap_in_entry),
#             }, 0)
#         map_entry, _ = map_collapse.apply(graph, sdfg)

#         map_fusion = MapFusion()
#         map_fusion.setup_match(sdfg, self.sdfg_id, self.state_id, {
#             MapFusion.first_map_exit: graph.node_id(self.tmap_exit),
#             MapFusion.second_map_entry: graph.node_id(map_entry),
#         }, 0)
#         map_fusion.apply(graph, sdfg)
