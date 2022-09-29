
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

sdfg.expand_library_nodes()
sdfg.apply_transformations_repeated(MapExpansion)

#-----------------------------------------------------------
# Commenting out the following gives us a previous state of the SDFG
# It is in here to make working with mapFusion easier

StateFusion.apply_to(sdfg,
                     first_state = multiplication_state,
                     second_state = reduction_state)
sdfg.view()
sdfg.apply_transformations_repeated(StateFusion)

sdfg.apply_transformations(InlineSDFG)
inlines = sdutil.inline_sdfgs(sdfg)
print (inlines)
sdfg.view()

#-----------------------------------------------------------
# Here we try to apply the auto-tiling

state = next(n for n in sdfg.states() if n.label == 'Mult_state')

mult_map = next(n for n in state.nodes() if isinstance(n, dace.nodes.MapEntry) and n.label == 'Mult_maps')
GPU_tiling_transformation.GPU_Tiling.apply_to(sdfg, map_entry= mult_map)

#-----------------------------------------------------------
# Merging the Maps
 
state = next(n for n in sdfg.states() if n.label == 'Mult_state')

mult_map_exit = next(n for n in state.nodes() if isinstance(n, dace.nodes.MapExit) and n.label == 'GPU_map')
reduction_map_entry = next(n for n in state.nodes() if isinstance(n,dace.nodes.MapEntry) and n.label == 'Reduction_maps')

transient = next(aname for aname, desc in sdfg.arrays.items() if desc.transient)
access_node = next(n for n in state.nodes() if isinstance(n, dace.nodes.AccessNode) and n.data == transient)


from dace.sdfg.propagation import propagate_memlets_sdfg
propagate_memlets_sdfg(sdfg)

MapFusion.apply_to(sdfg,
                   first_map_exit = mult_map_exit,
                   array = access_node,
                   second_map_entry = reduction_map_entry)


mult_map_exit = next(n for n in state.nodes() if isinstance(n, dace.nodes.MapExit) and n.label == 'Block_map')
reduction_map_entry = next(n for n in state.nodes() if isinstance(n,dace.nodes.MapEntry) and n.label == 'Reduction_maps_j')

transient = next(aname for aname, desc in sdfg.arrays.items() if desc.transient and aname == '__s1_n19OUT_temp1_n20IN_temp1')
access_node = next(n for n in state.nodes() if isinstance(n, dace.nodes.AccessNode) and n.data == transient)

MapFusion.apply_to(sdfg,
                   first_map_exit = mult_map_exit,
                   array = access_node,
                   second_map_entry = reduction_map_entry)

mult_map_exit = next(n for n in state.nodes() if isinstance(n, dace.nodes.MapExit) and n.label == 'Mult_maps')
reduction_map_entry = next(n for n in state.nodes() if isinstance(n,dace.nodes.MapEntry) and n.label == 'gridSized_strides_map')

transient = next(aname for aname, desc in sdfg.arrays.items() if desc.transient and aname == '__s1_n3OUT_temp1_n17IN_temp1')
access_node = next(n for n in state.nodes() if isinstance(n, dace.nodes.AccessNode) and n.data == transient)

MapFusion.apply_to(sdfg,
                   first_map_exit = mult_map_exit,
                   array = access_node,
                   second_map_entry = reduction_map_entry)

sdfg.simplify()
#-----------------------------------------------------------

res = sdfg(A=A, B=B, __return=r, N=sz, MaxTs= blockDim*gridDim, BlockDim=blockDim, GridDim=gridDim, WarpSize= 32)


print(res)

print(checksum(A,B))
assert(np.allclose(res, checksum(A,B)))

print("all_close validated")
