import cupy
import numpy as np
import dace
from dace.sdfg import nodes
from dace.transformation.auto import auto_optimize as auto
from dace.transformation.dataflow import MapExpansion

def gaussInit(a):
    for i in range(a.size):
        a[i] = i % 131                  # The modulo ensures the numbers stay reasonably small and the prime makes accidental multiples a lot less likely

N = dace.symbol('N')
sz = 10000000



@dace.program
def inner_product_python(A: dace.float64[N], B: dace.float64[N]):
    return np.add.reduce(A * B)


sdfg = inner_product_python.to_sdfg(simplify=True)
for _, arr in sdfg.arrays.items():
    if not arr.transient:
        arr.storage = dace.StorageType.GPU_Global
sdfg.apply_gpu_transformations()
from map_reduce_gpu import MapReduceFusionGPU
sdfg.apply_transformations(MapReduceFusionGPU)
# auto.auto_optimize(sdfg, dace.DeviceType.GPU)


A = np.ones((sz))
gaussInit(A)
gA = cupy.asarray(A)
B = np.ones((sz))
gaussInit(B)
gB = cupy.asarray(B)

res = sdfg(A=gA, B=gB, N=sz, BlockDim=64, GridDim=65536)
print(res)
# print(np.add.reduce(A*B))

sdfg2 = inner_product_python.to_sdfg(simplify=True)
sdfg2._name += '_auto'
for _, arr in sdfg2.arrays.items():
    if not arr.transient:
        arr.storage = dace.StorageType.GPU_Global
auto.auto_optimize(sdfg2, dace.DeviceType.GPU)
res = sdfg2(A=gA, B=gB, N=sz)
print(res)

arows, acols = dace.symbol('arows'), dace.symbol('acols')
nnz, hcols = dace.symbol('nnz'), dace.symbol('hcols')

# @dace.program
# def ahht(rows: dace.int32[nnz], cols: dace.int32[nnz], h1: dace.float32[arows, hcols], h2: dace.float32[acols, hcols]):
#     out = np.ndarray((nnz,), dtype=np.float32)
#     for i in dace.map[0:nnz]:
#         r = rows[i]
#         c = cols[i]
#         out[i] = np.add.reduce(h1[r] * h2[c])
    
#     return out

# @dace.program
# def simpler_ahht(h1: dace.float32[nnz, hcols], h2: dace.float32[nnz, hcols], out: dace.float32[nnz]):
#     out[:] = np.add.reduce(h1 * h2, axis=1)
@dace.program
def simpler_ahht(h1: dace.float32[nnz, hcols], h2: dace.float32[nnz, hcols]):
    return np.add.reduce(h1 * h2, axis=1)

sdfg = simpler_ahht.to_sdfg(simplify=True)
for _, arr in sdfg.arrays.items():
    if not arr.transient:
        arr.storage = dace.StorageType.GPU_Global
sdfg.apply_gpu_transformations()
from map_reduce_gpu import MapReduceFusionGPU
sdfg.apply_transformations(MapReduceFusionGPU)

sa = 1000
snnz = 10000
sh = 1024

rows = []
for i in range(100):
    rows.extend([i]*10)
rows = np.array(rows, dtype=np.int32)

cols = []
for i in range(100):
    cols.extend([i for i in range(0, 100, 10)])
cols = np.array(cols, dtype=np.int32)

# h1 = np.ones((sa, sh), dtype=np.float32)
# h2 = np.ones((sa, sh), dtype=np.float32)
h1 = np.ones((snnz, sh), dtype=np.float32)
h2 = np.ones((snnz, sh), dtype=np.float32)
out = np.empty((snnz, ), dtype=np.float32)
gh1 = cupy.asarray(h1)
gh2 = cupy.asarray(h2)
gout = cupy.asarray(out)

# res = sdfg(rows, cols, h1, h2)
res = sdfg(h1=gh1, h2=gh2, out=gout, nnz=snnz, hcols=sh, BlockDim=64, GridDim=65536)
# assert np.allclose(gout, np.array([sh] * snnz, dtype=np.float32))
# for i in range(snnz):
#     if gout[i] != np.float32(sh):
#         print(i)
print(res[:10])

sdfg2 = simpler_ahht.to_sdfg(simplify=True)
sdfg2._name += '_auto'
for _, arr in sdfg2.arrays.items():
    if not arr.transient:
        arr.storage = dace.StorageType.GPU_Global
auto.auto_optimize(sdfg2, dace.DeviceType.GPU)
res = sdfg2(h1=gh1, h2=gh2, out=gout, nnz=snnz, hcols=sh)
print(res[:10])


# sdfg2 = dace.SDFG('reduction')
# sdfg2.add_array('A', shape=[N], dtype=dace.float64, storage=dace.StorageType.GPU_Global)
# sdfg2.add_array('B', shape=[N], dtype=dace.float64, storage=dace.StorageType.GPU_Global)
# sdfg2.add_array('__return', shape=[1], dtype=dace.float64, storage=dace.StorageType.GPU_Global)


# def KernelCall(state):

#     tasklet_code = '''
#     if (i + j == 0) {
#         out[0] = double(0);
#     }
#     double sum = double(0);
#     for (int id = i * 1024 + j; id < N; id += blockDim.x * gridDim.x) {
#         sum += in1[id] * in2[id];
#     }
#     for (int offset = warpSize/2; offset > 0; offset /= 2) {
#         sum += __shfl_down_sync(0xFFFFFFFF, sum, offset);
#     }
#     if (j % warpSize == 0) {
#         atomicAdd(out, sum);
#     }
#     '''

#     tasklet, me, mx = state.add_mapped_tasklet(
#         name='callingKernel',
#         map_ranges={'i': '0:min(int_ceil(N, 1024), 2048)', 'j': '0:1024'},            # i is blockIdx.x and j is threadIdx.x
#         inputs={'in1': dace.Memlet('A[0:N]'), 'in2': dace.Memlet('B[0:N]')},
#         outputs={'out': dace.Memlet('__return[0]')},
#         code=tasklet_code,
#         language=dace.dtypes.Language.CPP,
#         external_edges=True
#     )
#     out_conn = {'out': dace.pointer(dace.float64)}
#     tasklet.out_connectors = out_conn

#     me.map.schedule = dace.dtypes.ScheduleType.GPU_Device

# callState = sdfg2.add_state()
# KernelCall(callState)

# sdfg2.apply_transformations_repeated(MapExpansion)

# for n in callState.nodes():
#     if isinstance(n, nodes.MapEntry) and "j" in n.map.params:
#         n.map.schedule = dace.dtypes.ScheduleType.GPU_ThreadBlock
        
# res2 = sdfg2(A=gA, B=gB, N=sz)
# print(res2)

# assert(np.allclose(res, res2))
