import dace
import cupy
from dace.dtypes import ScheduleType
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
        
        #NOTE: old_para should not be __i or __j
        
        # NOTE: Once more I am going to assume BlockDim, GridDim and MaxTs are symbols in the SDFG!
        # NOTE: how do I make sure I don't end up with aliasing?
        
        # figure out where N is:
        
        para_N = map_entry.range[0][1]
        
        # print(para_N[0][1])
        # print(type(para_N[0][1]))

        for s in ('BlockDim', 'GridDim'):
            try:
                sdfg.add_symbol(s, dace.int32)
            except FileExistsError:
                pass

        new_i = '__i' + old_para if old_para == '__i' else '__i'
        new_j = '__j' + old_para if old_para == '__j' else '__j'
        
        current_map = map_entry.map
        current_map.range = subsets.Range([(new_i +'*BlockDim+' + new_j, str(para_N), 'BlockDim * GridDim')])
        current_map.schedule = dace.ScheduleType.Sequential
        
        para_N += 1
        
        i_map = nodes.Map(label ='GPU_map_i',
                          ndrange = subsets.Range([('0', 'Min(GridDim, int_ceil(' + str(para_N) +',BlockDim))-1', '1')]),
                          params=[new_i],
                          schedule= dace.dtypes.ScheduleType.GPU_Device)
        
        j_map = nodes.Map(label= 'Block_map_j',
                          ndrange= subsets.Range([('0', 'BlockDim-1' , '1')]),
                          params=[new_j],
                          schedule= dace.dtypes.ScheduleType.GPU_ThreadBlock)

        i_entry = nodes.MapEntry(i_map)
        j_entry = nodes.MapEntry(j_map)
        
        i_exit = nodes.MapExit(i_map)
        j_exit = nodes.MapExit(j_map)
        
        for edge in graph.out_edges(map_entry):
            # graph.remove_edge_and_connectors(edge)
            # conn_name = 'IN_' + edge.src.label
            # i_entry.add_in_connector(conn_name)
            src = graph.memlet_path(edge)[0].src
            src_conn = graph.memlet_path(edge)[0].src_conn
            graph.add_memlet_path(
                              src,
                              i_entry,
                              j_entry,
                              map_entry,
                              edge.dst,
                              memlet=edge.data,
                              src_conn=src_conn,
                              dst_conn=edge.dst_conn)
            graph.remove_memlet_path(edge)
        
        for edge in graph.in_edges(map_exit):
            dst = graph.memlet_path(edge)[-1].dst
            dst_conn = graph.memlet_path(edge)[-1].dst_conn
            graph.add_memlet_path(
                                edge.src,
                                map_exit,
                                j_exit,
                                i_exit,
                                dst,                                      
                                src_conn=edge.src_conn,
                                memlet= edge.data,
                                dst_conn=dst_conn)
            graph.remove_memlet_path(edge)

        # sdfg.view()
            
        
        return [map_entry] + [i_entry, j_entry]


