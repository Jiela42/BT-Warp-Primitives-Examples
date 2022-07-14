from atexit import register
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
sz = 80000

blockDim = 256
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
gaussInit(A)
gA = cupy.asarray(A)
B = np.ones((sz))
gaussInit(B)
gB = cupy.asarray(B)

r = cupy.array([float(0)])
"""
res = sdfg(A=gA, B=gB, N=sz)
print(res)
"""

MaxTs = dace.symbol('MaxTs')
GridDim = dace.symbol('GridDim')
BlockDim = dace.symbol('BlockDim')


sdfg2 = dace.SDFG('reduction')
sdfg2.add_array('A', shape=[N], dtype=dace.float64, storage=dace.StorageType.GPU_Global)
sdfg2.add_array('B', shape=[N], dtype=dace.float64, storage=dace.StorageType.GPU_Global)
sdfg2.add_array('__return', shape=[1], dtype=dace.float64, storage=dace.StorageType.GPU_Global)
sdfg2.add_symbol('MaxTs', stype= dace.int32)
sdfg2.add_symbol('GridDim', stype= dace.int32)
sdfg2.add_symbol('BlockDim', stype= dace.int32)

def InitOut(state, i_out):
    
    dst_node = state.add_write(i_out)
    
    tasklet = state.add_tasklet('init_out', {}, {'out'}, 'out = double(0)')
    state.add_edge(tasklet, None, dst_node, 'out', memlet= dace.Memlet(data= i_out, subset = '0'))

def GridStrideLoop(state, i1, i2, mS):
    
    src_A = state.add_read(i1)
    src_B = state.add_read(i2)
    
    dst_node = state.add_write(mS)
    
    me,mx = state.add_map('gridSized_strides_map', dict(tId = 'i*BlockDim+j:N:tId*MaxTs'))
    tasklet = state.add_tasklet('mult', {'in1', 'in2'}, {'out'},  'out += in1 * in2')
    
    state.add_memlet_path(src_A, me, tasklet, dst_conn='in1', memlet=dace.Memlet(data=i1, subset='tId'))
    state.add_memlet_path(src_B, me, tasklet, dst_conn='in2', memlet=dace.Memlet(data=i2, subset = 'tId'))
    state.add_memlet_path(tasklet, mx, dst_node, src_conn='out', memlet=dace.Memlet(data=mS, subset='0'))
    
def WarpWiseReduction(state):
    tasklet_code = '''
    int offset = 1 << k;
    accOut += __shfl_down_sync(0xFFFFFFFF, accOut, offset);
    '''
    
    # for loop
    # hardcode entire loop
    
    tasklet, me, mx = state.add_mapped_tasklet(
        name = 'warpwise_Reduction',
        map_ranges={'k':'5:0'},
        inputs= {'accIn': dace.Memlet.from_array('mySum',state.parent.arrays['mySum'])},
        outputs= {'accOut': dace.Memlet.from_array('mySum', state.parent.arrays['mySum'])},
        code=tasklet_code,
        language=dace.dtypes.Language.CPP,
        external_edges= True
    )
    

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
WarpWiseReduction(wwr_state)


gpu_sdfg.add_edge(m_state, wwr_state, dace.InterstateEdge())

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
        


res2 = sdfg2(A=gA, B=gB, __return=r, N=sz, MaxTs = blockDim * gridDim, BlockDim = blockDim, GridDim = gridDim)
print(res2)

#assert(np.allclose(res, res2))
