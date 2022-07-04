import cupy
import numpy as np
import dace
from dace.sdfg import nodes
from dace.transformation.auto import auto_optimize as auto
from dace.transformation.dataflow import MapExpansion

# I keep hardcoding the maxNumber of threads, we might wanna add a symbol instead

def gaussInit(a):
    for i in range(a.size):
        a[i] = i % 131          # The modulo ensures the numbers stay reasonably small and the prime makes accidental multiples a lot less likely

N = dace.symbol('N')
sz = 80000



@dace.program
def inner_product_python(A: dace.float64[N], B: dace.float64[N]):
    return np.add.reduce(A * B)


sdfg = inner_product_python.to_sdfg(simplify=True)
for _, arr in sdfg.arrays.items():
    if not arr.transient:
        arr.storage = dace.StorageType.GPU_Global
auto.auto_optimize(sdfg, dace.DeviceType.GPU)


A = np.ones((sz))
gaussInit(A)
gA = cupy.asarray(A)
B = np.ones((sz))
gaussInit(B)
gB = cupy.asarray(B)

res = sdfg(A=gA, B=gB, N=sz)
print(res)

sdfg2 = dace.SDFG('reduction')
sdfg2.add_array('A', shape=[N], dtype=dace.float64, storage=dace.StorageType.GPU_Global)
sdfg2.add_array('B', shape=[N], dtype=dace.float64, storage=dace.StorageType.GPU_Global)
sdfg2.add_array('__return', shape=[1], dtype=dace.float64, storage=dace.StorageType.GPU_Global)

def GridStrideLoop(state, i1, i2, m_out):
    
    src_A = state.add_read(i1)
    src_B = state.add_read(i2)
    
    dst_node = state.add_write(m_out)
    
    me,mx = state.add_map('gridSized_strides_map', dict(tId = '0:min(1,256*2048/N)'))
    tasklet = state.add_tasklet('mult', {'in1', 'in2'}, {'out'},  'out += in1[tId] * in2[tId]')
    
    state.add_memlet_path(src_A, me, tasklet, dst_conn='in1', memlet=dace.Memlet(data=A, subset='i* 256 + j + k * (256*2048)'))
    state.add_memlet_path(src_B, me, tasklet, dst_conn='in2', memlet=dace.Memlet(data=B, subset = 'i* 256 + j + k * (256*2048)'))
    state.add_memlet_path(tasklet, mx, dst_node, src_conn='out', memlet=dace.Memlet(data=m_out, subset='k'))
    
    
    
    """
    
    tasklet, me, mx = state.add_mapped_tasklet(
        name= 'GridStrideLoop',
        map_ranges={'tId' : 'range(i * 256 + j, N, blockDim.x*gridDim.x)'},
        inputs= {'in1': dace.Memlet('A[0:N]'), 'in2': dace.Memlet('B[0:N]')},
        outputs={'out': dace.Memlet('multiplied[0:min(N,256*2048)')},
        code = ''' out += in1[tId] * in2[tId]; '''
        language= dace.dtypes.Language.CPP,
        external_edges=True)
    out_conn = {'out': dace.pointer(dace.float64)}
    
    tasklet.out_connectors=  out_conn
    me.map.schedule = dace.dtypes.ScheduleType.GPU_Device
    
    """
    

    

def GPUCall(state):

    tasklet_code = '''
    if (i + j == 0) {
        out[0] = double(0);
    }
    double sum = double(0);
    for (int id = i * 256 + j; id < N; id += blockDim.x * gridDim.x) {
        sum += in1[id] * in2[id];
    }
    for (int offset = warpSize/2; offset > 0; offset /= 2) {
        sum += __shfl_down_sync(0xFFFFFFFF, sum, offset);
    }
    if (j % warpSize == 0) {
        atomicAdd(out, sum);
    }
    '''

    tasklet, me, mx = state.add_mapped_tasklet(
        name='GPUcallingKernel',
        map_ranges={'i': '0:min(int_ceil(N, 256), 2048)', 'j': '0:256'},            # i is blockIdx.x and j is threadIdx.x
        inputs={'in1': dace.Memlet('A[0:N]'), 'in2': dace.Memlet('B[0:N]')},
        outputs={'out': dace.Memlet('__return[0]')},
        code=tasklet_code,
        language=dace.dtypes.Language.CPP,
        external_edges=True
    )
    out_conn = {'out': dace.pointer(dace.float64)}
    tasklet.out_connectors = out_conn

    me.map.schedule = dace.dtypes.ScheduleType.GPU_Device

gpuCallState = sdfg2.add_state()
GPUCall(gpuCallState)

gpu_sdfg = dace.SDFG('GPU_SDFG')
gpu_sdfg.add_array('multiplied', shape=[min(sz, 256*2048)], dtype=dace.float64, storage=dace.StorageType.GPU_Shared)

m_state= gpu_sdfg.add_state('Mult')
GridStrideLoop(m_state, 'A', 'B', 'multiplied')


da_whole_SDFG = gpuCallState.add_nested_sdfg(gpu_sdfg, sdfg2, {'A', 'B'}, {'return or something'})

sdfg2.apply_transformations_repeated(MapExpansion)

for n in gpuCallState.nodes():
    if isinstance(n, nodes.MapEntry) and "j" in n.map.params:
        n.map.schedule = dace.dtypes.ScheduleType.GPU_ThreadBlock
        
res2 = sdfg2(A=gA, B=gB, N=sz)
print(res2)

assert(np.allclose(res, res2))
