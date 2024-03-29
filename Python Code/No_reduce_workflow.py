
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
from dace.libraries.standard.nodes import Reduce
import GPU_tiling_transformation
from dace.sdfg import utils as sdutil
import numpy as np
import cupy
from dace.transformation import helpers
from dace.transformation.passes import pattern_matching as pm


def gaussInit(a):
    for i in range(a.size):
        a[i] = i % 131

def checksum(A,B):
    out = float(0)
    for i in range(A.size):
        out += A[i] *B[i]
    return out


N = dace.symbol('N')
sz = 10000

blockDim = 256
gridDim = 2048

A1 = np.ones(sz)
gaussInit(A1)
B1 = np.ones(sz)
gaussInit(B1)

A = cupy.asarray(A1)
B = cupy.asarray(B1)

r = cupy.array([float(0)])

MaxTs = dace.symbol('MaxTs')
GridDim = dace.symbol('GridDim')
BlockDim = dace.symbol('BlockDim')
WarpSize = dace.symbol('WarpSize')

sdfg = dace.SDFG('reduction')
sdfg.add_array('A', shape=[N], dtype=dace.float64)
sdfg.add_array('B', shape=[N], dtype=dace.float64)
sdfg.add_array('temp1', shape=[N], dtype=dace.float64, transient=True)
sdfg.add_array('__return', shape=[1], dtype=dace.float64)
sdfg.add_symbol('MaxTs', stype= dace.int32)
sdfg.add_symbol('GridDim', stype= dace.int32)
sdfg.add_symbol('BlockDim', stype= dace.int32)
sdfg.add_symbol('Warpsize', stype= dace.int32)

#-----------------------------------------------------------
# initialize Return

def Init(state, m):
    dst_node = state.add_write(m)
    init_t = state.add_tasklet('init_out', {}, {'__out'}, '__out = 0')
    
    state.add_edge (init_t, '__out', dst_node, None, dace.Memlet(data=m))

init_state = sdfg.add_state()
Init(init_state, '__return')

#-----------------------------------------------------------
# The Multiplication

def multiply(state, inA, inB, temp):
    node_A = state.add_read(inA)
    node_B = state.add_read(inB)
    node_tOut = state.add_write(temp)

    m_me,m_mx = state.add_map('Mult_maps', dict(tId='0:N'))

    mtasklet = state.add_tasklet('mult', {'in1', 'in2'}, {'out'}, 'out = in1 * in2')

    state.add_memlet_path(node_A, m_me, mtasklet, dst_conn = 'in1', memlet=dace.Memlet(data='A', subset='tId'))
    state.add_memlet_path(node_B, m_me, mtasklet, dst_conn='in2', memlet=dace.Memlet(data='B', subset='tId'))
    state.add_memlet_path(mtasklet, m_mx, node_tOut, src_conn= 'out', memlet=dace.Memlet(data='temp1', subset= 'tId'))

multiplication_state = sdfg.add_state('Mult_state')
multiply(multiplication_state, 'A', 'B', 'temp1')
#-----------------------------------------------------------
# The Reduction part (nested SDFG)
# This simulates the controlflow that we expect to see on a GPU (rTemp acts as array containing all mySum)

main_reduction_state = sdfg.add_state('Outer_reduction_state')
r_me, r_mx = main_reduction_state.add_map('Reduction_maps',  {'i': '0:min(int_ceil(N, BlockDim), GridDim)', 'j': '0:BlockDim'}) 

reduction_sdfg = dace.SDFG('Reduction_SDFG')

reduction_sdfg.add_array('rIn', shape=[N], dtype=dace.float64, storage=StorageType.GPU_Global)
reduction_sdfg.add_array('rSum', shape=[1], dtype=dace.float64, transient=True, storage=StorageType.Register)
reduction_sdfg.add_array('rOut', shape=[1], dtype=dace.float64, storage=StorageType.GPU_Global)
        
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

#-----------------------------------------------------------
# Connecting the nested SDFG
nsdfg = main_reduction_state.add_nested_sdfg(reduction_sdfg, sdfg, {'rIn'}, {'rOut'})

node_temp1 = main_reduction_state.add_read('temp1')
node_return = main_reduction_state.add_access('__return')

main_reduction_state.add_memlet_path(node_temp1, r_me, nsdfg, memlet=dace.Memlet(data='temp1', subset='0: (min(int_ceil(N, BlockDim), GridDim) * BlockDim)'), dst_conn='rIn')
main_reduction_state.add_memlet_path(nsdfg, r_mx, node_return, memlet=dace.Memlet(data='__return', subset='0'), src_conn= 'rOut')

sdfg.add_edge(init_state, multiplication_state, InterstateEdge())
sdfg.add_edge(multiplication_state, main_reduction_state, InterstateEdge())

sdfg.apply_transformations_repeated(MapExpansion)

#-----------------------------------------------------------
# Here we try to apply the auto-tiling

state = next(n for n in sdfg.states() if n.label == 'Mult_state')

mult_map = next(n for n in state.nodes() if isinstance(n, dace.nodes.MapEntry) and n.label == 'Mult_maps')
GPU_tiling_transformation.GPU_Tiling.apply_to(sdfg, map_entry= mult_map)

sdfg.view()

#-----------------------------------------------------------
# This part inlines the reduction, merges state and overall simplifies the SDFG

StateFusion.apply_to(sdfg,
                     first_state = multiplication_state,
                     second_state = main_reduction_state)

sdfg.apply_transformations_repeated(StateFusion)

sdfg.apply_transformations(InlineSDFG)

#-----------------------------------------------------------
# Putting it onto the GPU

for _, arr in sdfg.arrays.items():
        if not arr.transient:
            arr.storage = dace.StorageType.GPU_Global

for state in sdfg.states():
    for n in state.nodes():
        # TODO: find a way to un-parametrize this, such that it can be applied before Auto-tile as well!!
        if isinstance(n, nodes.MapEntry) and "__i" in n.map.params:
            n.map.schedule = dace.dtypes.ScheduleType.GPU_Device

for state in sdfg.states():
    for n in state.nodes():
        if isinstance(n, nodes.MapEntry) and "__j" in n.map.params:
            n.map.schedule = dace.dtypes.ScheduleType.GPU_ThreadBlock

#-----------------------------------------------------------
# Fusing the maps

pattern = sdutil.node_path_graph(dace.nodes.MapExit, dace.nodes.AccessNode, dace.nodes.MapEntry)

# sdfg.view()

while(len(list(pm.enumerate_matches(sdfg, pattern)))> 0):
    subgraph = next(n for n in pm.enumerate_matches(sdfg, pattern))
    # print("Current Match", subgraph.graph.label, ". Nodes:", subgraph.nodes())

    mult_map_exit = next(n for n in subgraph.nodes() if isinstance(n, dace.nodes.MapExit))
    reduction_map_entry = next(n for n in subgraph.nodes() if isinstance(n,dace.nodes.MapEntry))

    access_node = next(n for n in subgraph.nodes() if isinstance(n, dace.nodes.AccessNode))

    from dace.sdfg.propagation import propagate_memlets_sdfg
    propagate_memlets_sdfg(sdfg)

    MapFusion.apply_to(sdfg,
                    first_map_exit = mult_map_exit,
                    array = access_node,
                    second_map_entry = reduction_map_entry)

sdfg.simplify()
sdfg.apply_gpu_transformations()


#-----------------------------------------------------------

res = sdfg(A=A, B=B, __return=r, N=sz, MaxTs= blockDim*gridDim, BlockDim=blockDim, GridDim=gridDim, WarpSize= 32)


print(res)

print(checksum(A,B))
assert(np.allclose(res, checksum(A,B)))

print("all_close validated")
