import dace
import cupy
from dace.symbolic import int_ceil
import numpy as np

from dace.sdfg import nodes, SDFG, SDFGState
from dace.transformation import transformation as xf
from dace.sdfg import utils as sdutil
from dace import subsets

class GPU_Tiling(xf. SingleStateTransformation):
    
    map_entry = xf.PatternNode(dace.nodes.MapEntry)
    
    @classmethod
    def expressions(cls):
        return [sdutil.node_path_graph(cls.map_entry)]
    
    def can_be_applied(self, graph: SDFGState, expr_index: int, sdfg: SDFG, permissive: bool = False) -> bool:
        
        # for now we only support tiling for one entry!
        if (len(self.map_entry.params) > 1):
            return False
        else:
            return True
    
    def apply(self, graph: subsets.Union[SDFG, SDFGState], sdfg: SDFG):
        # we expand it to i,j, 'old_para', where i is DeviceMapped, j is the block and old_para is anything that is left over
        
        # remember to connect all the edges... or can you just change the range of old_para?
        map_entry = self.map_entry
        map_exit = graph.exit_node(map_entry)
        old_para = self.map_entry.params[0]
        
        # NOTE: Once more I am going to assume BlockDim, GridDim and MaxTs are symbols in the SDFG!
        # NOTE: how do I make sure I don't end up with aliasing?
        
        # figure out where N is:
        
        para_N = map_entry.range[0][1]
        
        # print(para_N[0][1])
        # print(type(para_N[0][1]))

        current_map = map_entry.map
        current_map.range = subsets.Range([('i*BlockDim+j', str(para_N), 'MaxTs')])
        
        para_N += 1
        
        i_map = nodes.Map(label ='GPU_map',
                          ndrange = subsets.Range([('0', 'Min(GridDim, int_ceil(' + str(para_N) +',BlockDim))-1', '1')]),
                          params=["i"])
        
        j_map = nodes.Map(label= 'Block_map',
                          ndrange= subsets.Range([('0', 'BlockDim-1' , '1')]),
                          params=['j'])

        i_entry = nodes.MapEntry(i_map)
        j_entry = nodes.MapEntry(j_map)
        
        i_exit = nodes.MapExit(i_map)
        j_exit = nodes.MapExit(j_map)
        
        for edge in graph.in_edges(map_entry):
            graph.remove_edge_and_connectors(edge)
            # conn_name = 'IN_' + edge.src.label
            # i_entry.add_in_connector(conn_name)
            graph.add_memlet_path(edge.src,
                              i_entry,
                              j_entry,
                              map_entry,
                              memlet=edge.data,
                              src_conn= edge.src_conn,
                              dst_conn = edge.dst_conn)
        
        # for edge in graph.out_edges(map_entry):
        #     graph.add_memlet_path(i_entry,
        #                           j_entry,
        #                           map_entry,
        #                           src_conn=edge.src_conn,
        #                           memlet= edge.data,
        #                           dst_conn=edge.dst_conn)
        
        # for edge in graph.in_edges(map_exit):
            
        
        for edge in graph.out_edges(map_exit):
            graph.remove_edge(edge)
            graph.add_memlet_path(map_exit,
                                j_exit,
                                i_exit,
                                edge.dst,                                      
                                src_conn=edge.src_conn,
                                memlet= edge.data,
                                dst_conn=edge.dst_conn)
        
        return [map_entry] + [i_entry, j_entry]


