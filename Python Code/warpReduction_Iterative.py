
from math import floor
import numpy as np
import dace

 # Da goal: sum ai * bi

size = (1 << 10) + 99

a = np.ones((size), dtype=float)
b = np.ones((size), dtype=float)
res = np.zeros((size), dtype=float)

print(a)
print(b)
print(res)

res_vec = np.multiply(a,b)
checkSum = np.sum(res_vec);

print(checkSum)

Size = dace.symbol('Size')
Overshoot = dace.symbol('Overshoot')

sdfg = dace.SDFG('reduction')

sdfg.add_symbol('Size', stype=dace.int32)
sdfg.add_symbol('Overshoot', stype=dace.int32)
sdfg.add_array('A', shape=[size], dtype=dace.float64)
sdfg.add_array('B', shape=[size], dtype=dace.float64)

sdfg.add_array('Res', shape=[size], dtype=dace.float64)

floorPow = 1

while(floorPow < size):
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

def partitionState(state, A, res_low, res_high):

    src_A = state.add_read(A)

    dst_node_low = state.add_write(res_low)
    dst_node_high = state.add_write(res_high)

    me_low, mx_low = state.add_map ('low', dict(i = '0:s/2'))
    me_high, mx_high = state.add_map('high', dict(j = 's/2: s'))

    tasklet_low = state.add_tasklet('write_Low', {'in_low'}, {'out_low'}, 'out_low = in_low')
    tasklet_high = state.add_tasklet('write_High', {'in_high'}, {'out_high'}, 'out_high = in_high')

    state.add_memlet_path(src_A, me_low, tasklet_low, dst_conn = 'in_low', memlet=dace.Memlet(data=A, subset='i'))
    state.add_memlet_path(src_A, me_high, tasklet_high, dst_conn='in_high', memlet=dace.Memlet(data=A, subset='j'))

    state.add_memlet_path(tasklet_low, mx_low, dst_node_low, src_conn='out_low', memlet=dace.Memlet(data = res_low, subset='i'))
    state.add_memlet_path(tasklet_high, mx_high, dst_node_high, src_conn='out_high', memlet=dace.Memlet(data=res_high, subset='j-(s/2)'))

def reductionStep(state, A, B, r):
    src_A = state.add_read(A)
    src_B = state.add_read(B)

    dst_node = state.add_write(r)

    me, mx = state.add_map('reduceMap', dict(i = '0:s'))
    tasklet = state.add_tasklet('reduce', {'in1', 'in2'},{'out'}, 'out = in1 + in2')

    state.add_memlet_path(src_A, me, tasklet, dst_conn='in1', memlet=dace.Memlet(data=A, subset='i'))
    state.add_memlet_path(src_B, me, tasklet, dst_conn='in2', memlet=dace.Memlet(data=B, subset = 'i'))
    state.add_memlet_path(tasklet, mx, dst_node, src_conn='out', memlet=dace.Memlet(data=r, subset='i'))

def report_end(state, r):
    r_node = state.add_read(r)
    t = state.add_tasklet('endtask', {'_in'}, {}, 'printf("done %f\\n", _in)')
    state.add_edge(r_node, None, t, '_in', dace.Memlet(data=r, subset='0'))

state0 = sdfg.add_state('Hoisted')
hoistedState(state0, 'A', 'B', 'Res')

# instead I might wanna use a fancy interstate edge that does some partitioning <3
state_SplitArr = sdfg.add_state('SplitArr')
partitionState(state_SplitArr, 'Res', 'A', 'B')

state_reductionStep = sdfg.add_state('addition')
reductionStep(state_reductionStep, 'A', 'B', 'Res')

guard = sdfg.add_state('guard')

end_state = sdfg.add_state()
report_end(end_state, 'Res')

sdfg.add_edge(state0, guard, dace.InterstateEdge(assignments= dict(s='Size')))
sdfg.add_edge(guard, state_SplitArr, dace.InterstateEdge('s > 1'))
sdfg.add_edge(state_SplitArr, state_reductionStep, dace.InterstateEdge('s > 1', assignments= dict(s = 's/2')))
sdfg.add_edge(state_SplitArr, guard, dace.InterstateEdge('s <= 1'))
sdfg.add_edge(state_reductionStep, guard, dace.InterstateEdge())
sdfg.add_edge(guard, end_state, dace.InterstateEdge('s <= 1'))

sdfg.validate
sdfg.simplify

sdfg(A=a, B=b, Res=res, Size=size, Overshoot=overshoot)
print(a)
print(b)
#print(res)
#sdfg.validate()
print("Ended")