
from math import floor
import numpy as np
import dace

 # Da goal: sum ai * bi


"""
To Do list so I don't end up forgetting:
any jagged ends aka, non-power of two ends, shall be taken care of in the hoised state
to this end we would need to find the smallest power of two and add in anything that's bigger than that into say the first few elements
"""

# we start with the size = 32 = warpSize
size = 1 << 5

a = np.ones((size), dtype=float)
b = np.ones((size), dtype=float)
res = np.zeros((size), dtype=float)

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

"""
def hoistedState(state, A, B, r):

    src_A = state.add_read(A)
    src_B = state.add_read(B)

    dst_node = state.add_write(r)

    mme, mmx = state.add_map('multiplyAll', dict(i='0:Size+Overshoot'))
    tMult = state.add_tasklet('mult', {'in1', 'in2'}, {'out'}, 'out = in1 * in2')

    ame, amx = state.add_map('addOvershoot', dict(j = 'Size:Size+Overshoot'))
    tAdd = state.add_tasklet('add_overshoot', {'in11, in22'}, {'out'}, 'out = in11 + in22')

    state.add_memlet_path(src_A, mme, tMult, dst_conn='in1', memlet=dace.Memlet(data=A, subset='i'))
    state.add_memlet_path(src_B, mme, tMult, dst_conn='in2', memlet=dace.Memlet(data=B, subset = 'i'))
    state.add_memlet_path(tMult, mmx, dst_node, src_conn='out', memlet=dace.Memlet(data=r, subset='i'))

    state.add_memlet_path(dst_node, ame, tAdd, dst_conn='in11', memlet=dace.Memlet(data=r, subset = 'Size:Size + Overshoot'))
    state.add_memlet_path(dst_node, ame, tAdd, dst_conn='in22', memlet=dace.Memlet(data=r, subset = 'Size:Size + Overshoot'))
    state.add_memlet_path(tMult, amx, dst_node, src_conn='out', memlet=dace.Memlet(data=r, subset ='0:Overshoot'))
"""

"""
"""
def hoistedState(state, A, B, r):

    src_A = state.add_read(A)
    src_B = state.add_read(B)

    dst_node = state.add_write(r)

    mme, mmx = state.add_map('multiplyAll', dict(i='0:Size+Overshoot'))
    tMult = state.add_tasklet('mult', {'in1', 'in2'}, {'out'}, 'out = in1 * in2')

    state.add_memlet_path(src_A, mme, tMult, dst_conn='in1', memlet=dace.Memlet(data=A, subset='i'))
    state.add_memlet_path(src_B, mme, tMult, dst_conn='in2', memlet=dace.Memlet(data=B, subset = 'i'))
    state.add_memlet_path(tMult, mmx, dst_node, src_conn='out', memlet=dace.Memlet(data=r, subset='i'))

"""
def reduce_overshootState(state, r):
    node = state.add_access(r)

    me, mx = state.add_map ('AddOvershoot', dict(i = 'Size:Size+Overshoot'))
    tAdd = state.add_tasklet('addOvershoot', {'in1', 'in2'}, {'out'}, 'out = in1 + in2')
    
    state.add_memlet_path(node, me, tAdd, dst_conn='in1', memlet=dace.Memlet(data=r, subset= '0:Overshoot'))
    state.add_memlet_path(node, me, tAdd, dst_conn='in2', memlet=dace.Memlet(data=r, subset= 'Size:Size+Overshoot'))
    state.add_memlet_path(tAdd, mx, node, src_conn= 'out', memlet= dace.Memlet(data=r, subset= 'O:Overshoot'))

"""


def partitionState(state, A, res_low, res_high):

    src_A = state.add_read(A)

    dst_node_low = state.add_write(res_low)
    dst_node_high = state.add_write(res_high)

    me_low, mx_low = state.add_map ('low', dict(i = '0:s/2'))
    me_high, mx_high = state.add_map('high', dict(j = 's/2: s'))

    tasklet_low = state.add_tasklet('write_Low', {'_in'}, {'out_low'}, 'out_low = _in')
    tasklet_high = state.add_tasklet('write_High', {'_in'}, {'out_high'}, 'outhigh = _in')

    state.add_memlet_path(src_A, me_low, tasklet_low, dst_conn = '_in', memlet=dace.Memlet(data=A, subset='i'))
    state.add_memlet_path(src_A, me_high, tasklet_high, dst_conn='_in', memlet=dace.Memlet(data=A, subset='j'))

    state.add_memlet_path(tasklet_low, mx_low, dst_node_low, src_conn='out_low', memlet=dace.Memlet(data = res_low, subset='i'))
    state.add_memlet_path(tasklet_high, mx_high, dst_node_high, src_conn='out_high', memlet=dace.Memlet(data=res_high, subset='j'))

def reductionStep(state, A, B, r):
    src_A = state.add_read(A)
    src_B = state.add_read(B)

    dst_node = state.add_write(r)

    me, mx = state.add_map('reduceMap', dict(i = '0:s'))
    tasklet = state.add_tasklet('reduce', {'in1', 'in2'},{'out'}, 'out = in1 + in2')

    state.add_memlet_path(src_A, me, tasklet, dst_conn='in1', memlet=dace.Memlet(data=A, subset='i'))
    state.add_memlet_path(src_B, me, tasklet, dst_conn='in2', memlet=dace.Memlet(data=B, subset = 'i'))
    state.add_memlet_path(tasklet, mx, dst_node, src_conn='out', memlet=dace.Memlet(data=r, subset='i'))

def report_end(state):
    r = state.add_read('Res')
    t = state.add_tasklet('endtask', {'_in'}, {}, 'printf("done %f\\n", _in)')
    state.add_edge(r, None, t, '_in', dace.Memlet(data='Res', subset='0'))

state0 = sdfg.add_state('Hoisted')
hoistedState(state0, 'A', 'B', 'Res')

#state_overshoot = sdfg.add_state('Handle_Overshoot')
#reduce_overshootState(state_overshoot, 'Res')

# instead I might wanna use a fancy interstate edge that does some partitioning <3
state_SplitArr = sdfg.add_state('SplitArr')
partitionState(state_SplitArr, 'Res', 'A', 'B')

state_reductionStep = sdfg.add_state('addition')
reductionStep(state_reductionStep, 'A', 'B', 'Res')

guard = sdfg.add_state('guard')

end_state = sdfg.add_state()
report_end(end_state)


# sdfg.add_edge(state0, state_overshoot, dace.InterstateEdge())
# sdfg.add_edge(state_overshoot, guard, dace.InterstateEdge(assignments= dict(s='Size')))
sdfg.add_edge(state0, guard, dace.InterstateEdge(assignments= dict(s='Size')))
sdfg.add_edge(guard, state_SplitArr, dace.InterstateEdge())
sdfg.add_edge(state_SplitArr, state_reductionStep, dace.InterstateEdge('s > 1', assignments= dict(s = 's/2')))
sdfg.add_edge(state_reductionStep, guard, dace.InterstateEdge())
sdfg.add_edge(guard, end_state, dace.InterstateEdge('s == 1'))

sdfg.validate
sdfg.simplify

print(res[0])

sdfg(A=a, B=b, Res=res, Size=size, Overshoot=overshoot)
#sdfg.validate()
print("Ended")