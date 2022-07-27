import dace
from dace.transformation import transformation
from dace.sdfg import SDFG, SDFGState
import dace.library
from dace.libraries.standard.environments.cuda import CUDA


@dace.library.expansion
class WarpReductionExpansion(transformation.ExpandTransformation):
    
    
    
    environments = [CUDA]
    
    @staticmethod
    def expansion(node: SDFG.node, parent_state: SDFGState, parent_sdfg: SDFG, local_sum: SDFG.node, local_Result: SDFG.node):
        
        # ToDo: fix the data descriptors!
            # The data we recieve:
                # FROM PARENT
                # mySum
                # SRes
                # FROM PARENT'S PARENT
                # warpSize      this should be a free symbol
        
        # ToDo: implement checks (once this thing works when used appropriately)
        
        ###################
        # Grabbing the Data:
        
        input_edge = parent_state.in_edges(node)[0]
        output_edge = parent_state.out_edges(node)[0]
        
        input_data = parent_sdfg.arrays[input_edge.data]      # this I believe to be the location of sA, sB and mySum (the data) for now, this can probably be fixed by the way the library node is called (since sA and sB aren't needed)
        output_data = parent_sdfg.arrays[output_edge.data]     # and this the location of sRes
        
        in_ptr = dace.pointer(input_data.dtype)
        out_ptr = dace.pointer(output_data.dtype)
        
        
        ###################
        
        ###################
        # Notes to self:
        # 
        # I do not need an entry state
        # Writeback writes into sRes
        # Use pointers in Writeback
        # Can I access an access node of the parent sdfg (gpu_sdfg) in this state?
        # you need to do some connector renaming and stuff to actually get that data connected properly
        
        ###################
        
        def WriteBackState(state, mS, r):
    
            src_node = state.add_read(mS)
            dst_node = state.add_write(r)
            
            state.add_nedge(src_node, dst_node, dace.Memlet(f"{r}", wcr="lambda x, y: x + y"))      

        
        subSDFG = dace.SDFG('WarpReduction_SDFG')
        
        wwr_state= subSDFG.add_state('WarpWise_Reduction')

        mSum = wwr_state.add_access(local_sum)
        wtasklet = wwr_state.add_tasklet('warpwise_Reduction',
                                        {}, {'__out'},
                                        f"__out = __shfl_down_sync(0xFFFFFFFF, {mSum.data}, offset);",
                                        dace.Language.CPP)
        wwr_state.add_edge(wtasklet, '__out', local_sum, None, dace.Memlet(f"{local_sum.data}", wcr="lambda x, y: x + y"))

        _, _, after_state = subSDFG.add_loop(parent_state, wwr_state, None, 'offset', 'WarpSize / 2', 'offset > 0', 'offset / 2')

        write_back_state = subSDFG.add_state('Write_Back')
        WriteBackState(write_back_state, local_sum, local_Result)

        random_end_state = subSDFG.add_state('RandomEndState')

        subSDFG.add_edge(after_state, write_back_state, dace.InterstateEdge('j % WarpSize == 0'))
        subSDFG.add_edge(after_state, random_end_state, dace.InterstateEdge('j % WarpSize != 0'))
        subSDFG.add_edge(write_back_state, random_end_state, dace.InterstateEdge())
        
        ###################
        # Redefining outer connectors and add to node
        
        
        
        
        
        ###################
        
        return subSDFG

