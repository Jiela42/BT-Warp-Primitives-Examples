from atexit import register
from random import random
import cupy
from dace.dtypes import StorageType
import numpy as np
import dace
from dace.sdfg import nodes
from dace.transformation.auto import auto_optimize as auto
from dace.transformation.dataflow import MapExpansion
from sympy import true

# I keep hardcoding the maxNumber of threads, we might wanna add a symbol instead

def gaussInit(a):
    for i in range(a.size):
        a[i] = i % 131          # The modulo ensures the numbers stay reasonably small and the prime makes accidental multiples a lot less likely

N = dace.symbol('N')
sz = 64


# Careful: Number of threads aka BlockDim must be bigger than number of elements!
blockDim = 64
gridDim = 2048


"""

@dace.program
def inner_product_python(A: dace.float64[N], B: dace.float64[N]):
    return np.add.reduce(A * B)


sdfg = inner_product_python.to_sdfg(simplify=True)
for _, arr in sdfg.arrays.items():
    if not arr.transient:
        arr.storage = dace.StorageType.GPU_Global
auto.auto_optimize(sdfg, dace.DeviceType.GPU)
"""



A = np.ones((sz))
# gaussInit(A)
gA = cupy.asarray(A)
B = np.ones((sz))
# gaussInit(B)
gB = cupy.asarray(B)

r = cupy.array([float(0)])
"""
res = sdfg(A=gA, B=gB, N=sz)
print(res)
"""

MaxTs = dace.symbol('MaxTs')
GridDim = dace.symbol('GridDim')
BlockDim = dace.symbol('BlockDim')
WarpSize = dace.symbol('WarpSize')


sdfg2 = dace.SDFG('reduction')
sdfg2.add_array('A', shape=[N], dtype=dace.float64, storage=dace.StorageType.GPU_Global)
sdfg2.add_array('B', shape=[N], dtype=dace.float64, storage=dace.StorageType.GPU_Global)
sdfg2.add_array('__return', shape=[1], dtype=dace.float64, storage=dace.StorageType.GPU_Global)
sdfg2.add_symbol('MaxTs', stype= dace.int32)
sdfg2.add_symbol('GridDim', stype= dace.int32)
sdfg2.add_symbol('BlockDim', stype= dace.int32)
sdfg2.add_symbol('WarpSize', stype= dace.int32)

def InitOut(state, i_out):
    
    dst_node = state.add_write(i_out)
    
    tasklet = state.add_tasklet('init_out', {}, {'out'}, 'out = double(0)')
    state.add_edge(tasklet, None, dst_node, 'out', memlet= dace.Memlet(data= i_out, subset = '0'))

def GridStrideLoop(state, i1, i2, mS):

    init_tasklet = state.add_tasklet('init_sum', {}, {'__out'}, '__out = 0')
    sum_node = state.add_access(mS)
    state.add_edge(init_tasklet, '__out', sum_node, None, dace.Memlet.from_array(mS, state.parent.arrays[mS]))
    
    src_A = state.add_read(i1)
    src_B = state.add_read(i2)
    
    dst_node = state.add_access(mS)
    
    me,mx = state.add_map('gridSized_strides_map', dict(tId = 'i*BlockDim+j:N:MaxTs'))
    tasklet = state.add_tasklet('mult', {'in1', 'in2', '__in3'}, {'out'},  'out = in1 * in2 + __in3')
    
    state.add_memlet_path(src_A, me, tasklet, dst_conn='in1', memlet=dace.Memlet(data=i1, subset='tId'))
    state.add_memlet_path(src_B, me, tasklet, dst_conn='in2', memlet=dace.Memlet(data=i2, subset = 'tId'))
    state.add_memlet_path(sum_node, me, tasklet, dst_conn='__in3', memlet=dace.Memlet.from_array(mS, state.parent.arrays[mS]))
    state.add_memlet_path(tasklet, mx, dst_node, src_conn='out', memlet=dace.Memlet(data=mS, subset='0'))
    
def WarpWiseReduction(state):
    tasklet_code = '''
    int offset = 1 << (4-k);
    accIn += __shfl_down_sync(0xFFFFFFFF, accIn, offset);
    '''
    
    # for loop
    # hardcode entire loop
    
    tasklet, me, mx = state.add_mapped_tasklet(
        name = 'warpwise_Reduction',
        map_ranges={'k':'0:5'},
        inputs= {'accIn': dace.Memlet.from_array('mySum',state.parent.arrays['mySum'])},
        outputs= {'accOut': dace.Memlet.from_array('mySum', state.parent.arrays['mySum'])},
        code=tasklet_code,
        language=dace.dtypes.Language.CPP,
        external_edges= True
    )

def WriteBackState(state, mS, r):
    
    src_node = state.add_read(mS)
    dst_node = state.add_write(r)
    
    # You don't need a map here and the assumption is wrong. You need WCR for atomic writes.
    # In this case, you don't even need a tasklet, you can copy directly between the access nodes.
    # me, mx = state.add_map('Write_back', dict(wId = '0'))
    # tasklet = state.add_tasklet('Add', {'in1'}, {'out'}, 'out += in1')  #this assumes that dace's adds are atomic (and that += is associative)
    
    # state.add_memlet_path(src_node, me, tasklet, dst_conn= 'in1', memlet=dace.Memlet(data=mS, subset='0'))
    # state.add_memlet_path(tasklet, mx, dst_node, src_conn='out', memlet=dace.Memlet(data=r, subset='0')) 
    state.add_nedge(src_node, dst_node, dace.Memlet(f"{r}", wcr="lambda x, y: x + y"))      

##################################
# Create the GPU scheduleing state

gpuCallState = sdfg2.add_state()

me, mx = gpuCallState.add_map('GPU_map', {'i': '0:min(int_ceil(N, BlockDim), GridDim)', 'j': '0:BlockDim'})
me.map.schedule = dace.dtypes.ScheduleType.GPU_Device

##################################


##################################
# make the nested SDFG happen

gpu_sdfg = dace.SDFG('GPU_SDFG')

# gpu_sdfg.add_array('multiplied', shape=[min(sz, 256*2048)], dtype=dace.float64, storage=dace.StorageType.GPU_Global, transient=True)
gpu_sdfg.add_array('sA', shape= [N], dtype=dace.float64, storage=dace.StorageType.GPU_Global)
gpu_sdfg.add_array('sB', shape= [N], dtype= dace.float64, storage=dace.StorageType.GPU_Global)
gpu_sdfg.add_array('sRes', shape=[1], dtype=dace.float64, storage=StorageType.GPU_Global)
gpu_sdfg.add_scalar('mySum', dtype=dace.float64, storage=StorageType.Register, transient=True)

# gpu_sdfg.add_scalar('acc', dtype=dace.float64, storage=StorageType.Register, transient=True)
# create inner sdfg

m_state= gpu_sdfg.add_state('Mult')
GridStrideLoop(m_state, 'sA', 'sB', 'mySum')

wwr_state= gpu_sdfg.add_state('WarpWise_Reduction')
# WarpWiseReduction(wwr_state)
# tasklet_code = '''
#     int offset = 1 << (4-k);
#     accIn += __shfl_down_sync(0xFFFFFFFF, accIn, offset);
#     '''
    
#     # for loop
#     # hardcode entire loop
    
#     tasklet, me, mx = state.add_mapped_tasklet(
#         name = 'warpwise_Reduction',
#         map_ranges={'k':'0:5'},
#         inputs= {'accIn': dace.Memlet.from_array('mySum',state.parent.arrays['mySum'])},
#         outputs= {'accOut': dace.Memlet.from_array('mySum', state.parent.arrays['mySum'])},
#         code=tasklet_code,
#         language=dace.dtypes.Language.CPP,
#         external_edges= True
mSum = wwr_state.add_access('mySum')
wtasklet = wwr_state.add_tasklet('warpwise_Reduction',
                                 {}, {'__out'},
                                 f"__out = __shfl_down_sync(0xFFFFFFFF, {mSum.data}, offset);",
                                 dace.Language.CPP)
wwr_state.add_edge(wtasklet, '__out', mSum, None, dace.Memlet(f"{mSum.data}", wcr="lambda x, y: x + y"))

_, _, after_state = gpu_sdfg.add_loop(m_state, wwr_state, None, 'offset', 'WarpSize / 2', 'offset > 0', 'offset / 2')

write_back_state = gpu_sdfg.add_state('Write_Back')
WriteBackState(write_back_state, 'mySum', 'sRes')

random_end_state = gpu_sdfg.add_state('RandomEndState')


# gpu_sdfg.add_edge(m_state, wwr_state, dace.InterstateEdge())
# gpu_sdfg.add_edge(wwr_state, write_back_state, dace.InterstateEdge('j % WarpSize == 0'))
# gpu_sdfg.add_edge(wwr_state, random_end_state, dace.InterstateEdge('j % WarpSize != 0'))
gpu_sdfg.add_edge(after_state, write_back_state, dace.InterstateEdge('j % WarpSize == 0'))
gpu_sdfg.add_edge(after_state, random_end_state, dace.InterstateEdge('j % WarpSize != 0'))
gpu_sdfg.add_edge(write_back_state, random_end_state, dace.InterstateEdge())

##################################


##################################
# Make the dataflow between the states happen
da_whole_SDFG = gpuCallState.add_nested_sdfg(gpu_sdfg, sdfg2, {'sA', 'sB'}, {'sRes'})

Ain = gpuCallState.add_read('A')
Bin = gpuCallState.add_read('B')
ROut = gpuCallState.add_write('__return')

gpuCallState.add_memlet_path(Ain, me, da_whole_SDFG, memlet=dace.Memlet(data='A', subset='0: (min(int_ceil(N, BlockDim), GridDim) * BlockDim)'), dst_conn='sA')
gpuCallState.add_memlet_path(Bin, me, da_whole_SDFG, memlet=dace.Memlet(data='B', subset='0: (min(int_ceil(N, BlockDim), GridDim) * BlockDim)'), dst_conn='sB')
gpuCallState.add_memlet_path(da_whole_SDFG, mx, ROut, memlet=dace.Memlet(data='__return', subset='0'), src_conn='sRes')


sdfg2.apply_transformations_repeated(MapExpansion)

for n in gpuCallState.nodes():
    if isinstance(n, nodes.MapEntry) and "j" in n.map.params:
        n.map.schedule = dace.dtypes.ScheduleType.GPU_ThreadBlock
        


res2 = sdfg2(A=gA, B=gB, __return=r, N=sz, MaxTs = blockDim * gridDim, BlockDim = blockDim, GridDim = gridDim, WarpSize= 32)
print(res2)

#assert(np.allclose(res, res2))
