
import numpy as np
import dace

 # Da goal: sum ai * bi

# we start with the size = 32 = warpSize
size = 1 << 5

a = np.ones((1, size), dtype=float)
b = np.ones((1,size), dtype=float)
res = np.zeros((1,size), dtype=float)

res_vec = np.multiply(a,b)
checkSum = np.sum(res_vec);

# print(checkSum)

Size = dace.symbol('Size')

sdfg = dace.SDFG('reduction')

sdfg.add_symbol('Size', stype=dace.int32)
sdfg.add_array('A', shape=[size], dtype=float)
sdfg.add_array('B', shape=[size], dtype=float)

sdfg.add_array('Res', shape=[size], dtype=float)



def hoistedState(state, A, B, r):
    src_A = state.add_read(A)
    src_B = state.add_read(B)

    dst_node = state.add_write(r)

    me, mx = state.add_map('multiplyAll', dict(i='0:Size-1'))
    tasklet = state.add_tasklet('mult', {'A', 'B'}, {'r'}, 'r[i] = A[i] * B[i]')

    state.add_memlet_path(src_A, me, tasklet, dst_conn='A', memlet=dace.Memlet(data=A, subset='i'))
    state.add_memlet_path(src_B, me, tasklet, dst_conn='B', memlet=dace.Memlet(data=B, subset = 'i'))
    state.add_memlet_path(tasklet, mx, dst_node, src_conn='r', memlet=dace.Memlet(data=r, subset='i'))

state0 = sdfg.add_state()
hoistedState(state0, 'A', 'B', 'Res')

#def reductionState(state, A, B, r):

#def partitionState(state, r, r_high, r_low):

sdfg(A=a, B=b, Res=res, Size=size)
#sdfg.validate()
print("Ended")