from math import floor
import numpy as np
import dace
from dace.transformation.interstate import GPUTransformSDFG
from dace.transformation.interstate import StateFusion

# Da goal: sum ai * bi
# Specifically: put some of this into the gpu (just SOMETHING!!)

size = (1 << 5) + 3
a = np.ones((size), dtype=float)
b = np.ones((size), dtype=float)
res = np.zeros((size), dtype=float)

res_vec = np.multiply(a, b)
checkSum = np.sum(res_vec)

print(checkSum)

Size = dace.symbol('Size')
Overshoot = dace.symbol('Overshoot')

sdfg = dace.SDFG('reduction')

sdfg.add_symbol('Size', stype=dace.int32)
sdfg.add_symbol('Overshoot', stype=dace.int32)
sdfg.add_array('A', shape=[size], dtype=dace.float64)
sdfg.add_array('B', shape=[size], dtype=dace.float64)

sdfg.add_array('Res', shape=[size], dtype=dace.float64)

sdfg.add_transient('Temp', shape=[size], dtype=dace.float64)
sdfg.add_transient('gTemp', [size], dace.float64, dace.StorageType.GPU_Global)
sdfg.add_transient('gRes', [size], dace.float64, dace.StorageType.GPU_Global)

floorPow = 1

while (floorPow < size):
    floorPow = floorPow << 1

if floorPow > size:
    floorPow = floorPow >> 1

overshoot = size - floorPow
size = floorPow

def hoistedState(state, A, B, r):

    src_A = state.add_read(A)
    src_B = state.add_read(B)

    dst_node = state.add_write(r)

    mme1, mmx1 = state.add_map('Overshoot_to_size', dict(i='Overshoot:Size'))
    mme2, mmx2 = state.add_map('FirstOvershootElmts', dict(j='0:Overshoot'))

    tMult1 = state.add_tasklet('mult', {'in1', 'in2'}, {'out'}, 'out = in1 * in2')
    tMult_Add = state.add_tasklet('mult_and_add', {'in11', 'in12', 'in21', 'in22'}, {'out'}, 'out = in11 * in12 + in21 * in22')

    state.add_memlet_path(src_A, mme1, tMult1, dst_conn='in1', memlet=dace.Memlet(data=A, subset='i'))
    state.add_memlet_path(src_B, mme1, tMult1, dst_conn='in2', memlet=dace.Memlet(data=B, subset = 'i'))
    state.add_memlet_path(tMult1, mmx1, dst_node, src_conn='out', memlet=dace.Memlet(data=r, subset='i'))

    state.add_memlet_path(src_A, mme2, tMult_Add, dst_conn='in11', memlet=dace.Memlet(data=A, subset='j'))
    state.add_memlet_path(src_B, mme2, tMult_Add, dst_conn='in12', memlet=dace.Memlet(data=B, subset = 'j'))
    state.add_memlet_path(src_A, mme2, tMult_Add, dst_conn='in21', memlet=dace.Memlet(data=A, subset='j+Size'))
    state.add_memlet_path(src_B, mme2, tMult_Add, dst_conn='in22', memlet=dace.Memlet(data=B, subset = 'j+Size'))
    state.add_memlet_path(tMult_Add, mmx2, dst_node, src_conn='out', memlet=dace.Memlet(data=r, subset='j'))

def report_end(state, r):
    r_node = state.add_read(r)
    t = state.add_tasklet('endtask', {'_in'}, {}, 'printf("done %f\\n", _in)')
    state.add_edge(r_node, None, t, '_in', dace.Memlet(data=r, subset='0'))

def cub_state(state):
    tasklet = state.add_tasklet(
        name = 'cub_Node',
        inputs={'temp'},
        outputs={'res'},
        code = '''

        void *d_temp_storage= NULL;
        size_t temp_storage_bytes = 0;
        cub::DeviceReduce::Sum(d_temp_storage, temp_storage_bytes, temp, res, Size);

        cudaMalloc(&d_temp_storage, temp_storage_bytes);

        cub::DeviceReduce::Sum(d_temp_storage, temp_storage_bytes, temp, res, Size);
        ''',
        language=dace.Language.CPP
    )

    T = state.add_access('Temp')
    gT = state.add_access('gTemp')
    gR = state.add_access ('gRes')
    R = state.add_access('Res')

    state.add_edge(gT, None, tasklet,'temp', dace.Memlet('gTemp[0:Size]'))
    state.add_edge(tasklet, 'res', gR, None, dace.Memlet('gRes[0:Size]'))

    state.add_nedge(T, gT, dace.Memlet(data='Temp', subset='0:Size'))
    state.add_nedge(R, gR, dace.Memlet(data='Res', subset= '0:Size'))

state0 = sdfg.add_state('Hoisted')
hoistedState(state0, 'A', 'B', 'Temp')

cubState = sdfg.add_state('Cub_State')
cub_state(cubState)

end_state = sdfg.add_state()
report_end(end_state, 'Res')

sdfg.add_edge(state0, cubState, dace.InterstateEdge())
sdfg.add_edge(cubState, end_state, dace.InterstateEdge())

sdfg.append_global_code('''#include <cub/cub.cuh>''')

sdfg.validate
sdfg.simplify


sdfg(A=a, B=b, Res=res, Size=size, Overshoot=overshoot)

# sdfg.apply_transformations_repeated(StateFusion)
