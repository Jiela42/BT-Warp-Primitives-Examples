
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
from dace.transformation.interstate import multistate_inline
from dace.libraries.standard.nodes import Reduce
import GPU_tiling_transformation
from dace.sdfg import utils as sdutil
import numpy as np
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
sz = 10000000

blockDim = 256
gridDim = 2048

A = np.ones(sz)
gaussInit(A)
B = np.ones(sz)
gaussInit(B)

r = np.array([float(0)])

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
# sdfg.add_symbol('WarpSize', stype= dace.int32)
sdfg.add_constant('WarpSize', 32)

#-----------------------------------------------------------
# initialize Return

def Init(state, m):
    dst_node = state.add_write(m)
    init_t = state.add_tasklet('init_out', {}, {'__out'}, '__out = 0')
    
    state.add_edge (init_t, '__out', dst_node, None, dace.Memlet(data=m))

init_state = sdfg.add_state()
# Init(init_state, '__return')

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
reduction_state = sdfg.add_state()

aA = reduction_state.add_access('temp1')
aRet = reduction_state.add_access('__return')

red = Reduce(wcr = 'lambda a,b: a + b', identity = 0)
red.implementation = 'CUDA(shuffle)'

reduction_state.add_edge(aA, None, red, None, memlet= dace.Memlet.from_array('temp1', reduction_state.parent.arrays['temp1']))
reduction_state.add_edge(red, None, aRet, None, dace.Memlet.from_array('__return', reduction_state.parent.arrays['__return']))

#-----------------------------------------------------------
# Connecting the states

sdfg.add_edge(init_state, multiplication_state, InterstateEdge())
sdfg.add_edge(multiplication_state, reduction_state, InterstateEdge())

for _, arr in sdfg.arrays.items():
        if not arr.transient:
            arr.storage = dace.StorageType.GPU_Global
sdfg.arrays['temp1'].storage = dace.StorageType.GPU_Global

sdfg.apply_gpu_transformations()
# sdfg.view()

sdfg.expand_library_nodes()
sdfg.simplify()
sdfg.apply_transformations_repeated(MapExpansion)

#-----------------------------------------------------------
# from dace.transformation import helpers
# state = next(n for n in sdfg.nodes() if n.label == 'WarpReduction_SDFG')
# helpers.nest_sdfg_control_flow(state.sdfg)
#-----------------------------------------------------------

#-----------------------------------------------------------
# Here we try to apply the auto-tiling

# state = next(n for n in sdfg.states() if n.label == 'Mult_state')

# mult_map = next(n for n in state.nodes() if isinstance(n, dace.nodes.MapEntry) and n.label == 'Mult_maps')
mult_map = next(n for n, _ in sdfg.all_nodes_recursive() if isinstance(n, dace.nodes.MapEntry) and n.label == 'Mult_maps')
GPU_tiling_transformation.GPU_Tiling.apply_to(sdfg, map_entry= mult_map)

# Putting the data onto the GPU, the maps GPU tiling creates are on the gpu

# for _, arr in sdfg.arrays.items():
#         if not arr.transient:
#             arr.storage = dace.StorageType.GPU_Global
# sdfg.arrays['temp1'].storage = dace.StorageType.GPU_Global
#-----------------------------------------------------------
# This part inlines the reduction, merges state and overall simplifies the SDFG

# StateFusion.apply_to(sdfg,
#                      first_state = multiplication_state,
#                      second_state = reduction_state)

# sdfg.apply_transformations_repeated(StateFusion)

# inlines = sdfg.apply_transformations(multistate_inline.InlineMultistateSDFG)
# print(inlines)


# for state in sdfg.states():
#     for n in state.nodes():
#         # TODO: find a way to un-parametrize this, such that it can be applied before Auto-tile as well!!
#         if isinstance(n, nodes.MapEntry) and "__i" in n.map.params:
#             n.map.schedule = dace.dtypes.ScheduleType.GPU_Device

# for state in sdfg.states():
#     for n in state.nodes():
#         if isinstance(n, nodes.MapEntry) and "__j" in n.map.params:
#             n.map.schedule = dace.dtypes.ScheduleType.GPU_ThreadBlock

#-----------------------------------------------------------
# Fusing the maps
sdfg.view()
pattern = sdutil.node_path_graph(dace.nodes.MapExit, dace.nodes.AccessNode, dace.nodes.MapEntry)


while(len(list(pm.enumerate_matches(sdfg, pattern)))> 0):
    subgraph = next(n for n in pm.enumerate_matches(sdfg, pattern))
    print("Current Match", subgraph.graph.label, ". Nodes:", subgraph.nodes())

    mult_map_exit = next(n for n in subgraph.nodes() if isinstance(n, dace.nodes.MapExit))
    reduction_map_entry = next(n for n in subgraph.nodes() if isinstance(n,dace.nodes.MapEntry))

    access_node = next(n for n in subgraph.nodes() if isinstance(n, dace.nodes.AccessNode))

    from dace.sdfg.propagation import propagate_memlets_sdfg
    propagate_memlets_sdfg(sdfg)

    MapFusion.apply_to(sdfg,
                    first_map_exit = mult_map_exit,
                    array = access_node,
                    second_map_entry = reduction_map_entry)

    # sdfg.view()
sdfg.simplify()

# for sd in sdfg.all_sdfgs_recursive():
#     if sd.parent_sdfg is not None:
#         from dace.transformation.interstate import HoistState
#         HoistState.apply_to(sdfg, permissive=True, nsdfg=sd.parent_nsdfg_node)
#         break

sdfg.view()

#-----------------------------------------------------------
maxTs = min(blockDim*gridDim,sz)

import cupy
gA, gB, gr = (cupy.asarray(d) for d in (A, B, r))

res = sdfg(A=gA, B=gB, __return=gr, N=sz, MaxTs= maxTs, BlockDim = blockDim, GridDim = gridDim)


print(res)

print(checksum(A,B))
print (res)
print(checksum(A,B))
assert(np.allclose(res, checksum(A,B)))

print("all_close validated")
