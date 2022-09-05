
from cProfile import label
from multiprocessing import reduction
import dace
from dace import memlet
from dace.dtypes import StorageType
from dace.sdfg import nodes
from dace.sdfg.sdfg import InterstateEdge
from dace.transformation.dataflow import MapFusion
from dace.transformation.interstate import StateFusion
from dace.transformation.dataflow import OTFMapFusion
from dace.transformation.dataflow import MapTiling
from dace.transformation.dataflow import MapExpansion
from dace.transformation.interstate import InlineSDFG

import numpy as np


def gaussInit(a):
    for i in range(a.size):
        a[i] = i % 131

def checksum(A,B):
    out = float(0)
    for i in range(A.size):
        out += A[i] *B[i]
    return out


N = dace.symbol('N')
sz = 15

blockDim = 4
gridDim = 2

A = np.ones(sz)
gaussInit(A)
B = np.ones(sz)
gaussInit(B)

r = np.array([float(0)])

MaxTs = dace.symbol('MaxTs')
GridDim = dace.symbol('GridDim')
BlockDim = dace.symbol('BlockDim')

sdfg = dace.SDFG('reduction')
sdfg.add_array('A', shape=[N], dtype=dace.float64)
sdfg.add_array('B', shape=[N], dtype=dace.float64)
sdfg.add_array('temp1', shape=[N], dtype=dace.float64, transient=True)
sdfg.add_array('__return', shape=[1], dtype=dace.float64)
sdfg.add_symbol('MaxTs', stype= dace.int32)
sdfg.add_symbol('GridDim', stype= dace.int32)
sdfg.add_symbol('BlockDim', stype= dace.int32)

###################
# initialize Return

def Init(state, m):
    dst_node = state.add_write(m)
    init_t = state.add_tasklet('init_out', {}, {'__out'}, '__out = 0')
    
    state.add_edge (init_t, '__out', dst_node, None, dace.Memlet(data=m))

init_state = sdfg.add_state()
Init(init_state, '__return')


###################
# The Multiplication

def multiply(state, inA, inB, temp):
    node_A = state.add_read(inA)
    node_B = state.add_read(inB)
    node_tOut = state.add_write(temp)

    m_me,m_mx = state.add_map('tripple_map', {'i': '0:min(int_ceil(N, BlockDim), GridDim)', 'j': '0:BlockDim', 'tId': 'i*BlockDim+j:N:MaxTs'})

    mtasklet = state.add_tasklet('mult', {'in1', 'in2'}, {'out'}, 'out = in1 * in2')

    state.add_memlet_path(node_A, m_me, mtasklet, dst_conn = 'in1', memlet=dace.Memlet(data='A', subset='tId'))
    state.add_memlet_path(node_B, m_me, mtasklet, dst_conn='in2', memlet=dace.Memlet(data='B', subset='tId'))
    state.add_memlet_path(mtasklet, m_mx, node_tOut, src_conn= 'out', memlet=dace.Memlet(data='temp1', subset= 'tId'))

multiplication_state = sdfg.add_state('Mult_state')
multiply(multiplication_state, 'A', 'B', 'temp1')

###################
# The Reduction part (nested SDFG)
# This simulates the controlflow that we expect to see on a GPU (rTemp acts as array containing all mySum)

main_reduction_state = sdfg.add_state('Outer_reduction_state')
r_me, r_mx = main_reduction_state.add_map('Reduction_maps',  {'i': '0:min(int_ceil(N, BlockDim), GridDim)', 'j': '0:BlockDim'}) 

reduction_sdfg = dace.SDFG('Reduction_SDFG')

reduction_sdfg.add_array('rIn', shape=[N], dtype=dace.float64)
reduction_sdfg.add_array('rSum', shape=[1], dtype=dace.float64, transient=True, storage=StorageType.Register)
reduction_sdfg.add_array('rOut', shape=[1], dtype=dace.float64)
        
def Tile_and_Load(state, i1, mS):

    init_tasklet = state.add_tasklet('init_sum', {}, {'__out'}, '__out = 0')
    sum_node = state.add_access(mS)
    state.add_edge(init_tasklet, '__out', sum_node, None, dace.Memlet.from_array(mS, state.parent.arrays[mS]))
    
    src_A = state.add_read(i1)
    
    dst_node = state.add_access(mS)
    
    me,mx = state.add_map('gridSized_strides_map', dict(tId = 'i*BlockDim+j:N:MaxTs'))
    tasklet = state.add_tasklet('tiling', {'in1', '__in3'}, {'out'},  'out = in1 + __in3')
    
    state.add_memlet_path(src_A, me, tasklet, dst_conn='in1', memlet=dace.Memlet(data=i1, subset='tId'))
    state.add_memlet_path(sum_node, me, tasklet, dst_conn='__in3', memlet=dace.Memlet.from_array(mS, state.parent.arrays[mS]))
    state.add_memlet_path(tasklet, mx, dst_node, src_conn='out', memlet=dace.Memlet(data=mS, subset='0'))

def Reduce_and_Write_Back(state, mS, r):

    src_node = state.add_read(mS)
    dst_node = state.add_write(r)
    
    state.add_nedge(src_node, dst_node, dace.Memlet(f"{r}", wcr="lambda x, y: x + y"))      

tnl_state = reduction_sdfg.add_state('Tile_and_load')
reduction_state = reduction_sdfg.add_state('Reduction_state')

Tile_and_Load(tnl_state, 'rIn', 'rSum')
Reduce_and_Write_Back(reduction_state, 'rSum', 'rOut')

reduction_sdfg.add_edge(tnl_state, reduction_state, InterstateEdge())

###################
# Connecting the nested SDFG
nsdfg = main_reduction_state.add_nested_sdfg(reduction_sdfg, sdfg, {'rIn'}, {'rOut'})

node_temp1 = main_reduction_state.add_read('temp1')
node_return = main_reduction_state.add_access('__return')

main_reduction_state.add_memlet_path(node_temp1, r_me, nsdfg, memlet=dace.Memlet(data='temp1', subset='0: (min(int_ceil(N, BlockDim), GridDim) * BlockDim)'), dst_conn='rIn')
main_reduction_state.add_memlet_path(nsdfg, r_mx, node_return, memlet=dace.Memlet(data='__return', subset='0'), src_conn= 'rOut')

sdfg.add_edge(init_state, multiplication_state, InterstateEdge())
sdfg.add_edge(multiplication_state, main_reduction_state, InterstateEdge())

sdfg.apply_transformations_repeated(MapExpansion)

###################
# Commenting out the following gives us a previous state of the SDFG
# It is in here to make working with mapFusion easier

StateFusion.apply_to(sdfg,
                     first_state = multiplication_state,
                     second_state = main_reduction_state)

sdfg.apply_transformations_repeated(StateFusion)

sdfg.apply_transformations(InlineSDFG)



state = next(n for n in sdfg.states() if n.label == 'Mult_state')
###################

# sdfg.apply_transformations(MapTiling)

# MapTiling.apply_to(sdfg,
#                    first_map_exit = mult_map_exit,
#                    array = access_node,
#                    second_map_entry = reduction_map_entry)


mult_map_exit = next(n for n in state.nodes() if isinstance(n, dace.nodes.MapExit) and n.label == 'tripple_map')
reduction_map_entry = next(n for n in state.nodes() if isinstance(n,dace.nodes.MapEntry) and n.label == 'Reduction_maps')

transient = next(aname for aname, desc in sdfg.arrays.items() if desc.transient)
access_node = next(n for n in state.nodes() if isinstance(n, dace.nodes.AccessNode) and n.data == transient)


# MapFusion.apply_to(sdfg,
#                    first_map_exit = mult_map_exit,
#                    array = access_node,
#                    second_map_entry = reduction_map_entry)

# sdfg.apply_transformations_repeated(MapFusion)

res = sdfg(A=A, B=B, __return=r, N=sz, MaxTs= blockDim*gridDim, BlockDim=blockDim, GridDim=gridDim)


print(res)

print(checksum(A,B))

