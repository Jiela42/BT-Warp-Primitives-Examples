
import numpy as np
import dace

 # Da goal: sum ai * bi
"""
To do list, cause I will forget:
- Properly init bSizes!

"""


# we start with the size = 32 = warpSize
size = 1 << 5 + 2

a = np.ones((size), dtype=float)
b = np.ones((size), dtype=float)
res = np.zeros((size), dtype=float)
bSizes = np.array([32,2])

res_vec = np.multiply(a,b)
checkSum = np.sum(res_vec);

# print(checkSum)

# Size = dace.symbol('Size')

sdfg = dace.SDFG('full_Reduction')

# sdfg.add_symbol('Size', stype=dace.int32)
sdfg.add_scalar('dSize', dtype=dace.int32)
sdfg.add_scalar('numBlocks', dtype=dace.int64)
sdfg.add_array('blockSizes', dtype=dace.int32)
sdfg.add_array('A', shape=[size], dtype=dace.float64)
sdfg.add_array('B', shape=[size], dtype=dace.float64)

sdfg.add_array('Res', shape=[size], dtype=dace.float64)


def hoistedState(state, A, B, r):

    # size = state.add_access('dSize')

    src_A = state.add_read(A)
    src_B = state.add_read(B)

    dst_node = state.add_write(r)

    me, mx = state.add_map('multiplyAll', dict(i='0:dSize'))
    tasklet = state.add_tasklet('mult', {'in1', 'in2'}, {'out'}, 'out = in1 * in2')

    # me.add_in_connector('numElmts')

    #state.add_edge(size, None, me, 'numElmts', dace.Memlet('dSize'))

    state.add_memlet_path(src_A, me, tasklet, dst_conn='in1', memlet=dace.Memlet(data=A, subset='i'))
    state.add_memlet_path(src_B, me, tasklet, dst_conn='in2', memlet=dace.Memlet(data=B, subset = 'i'))
    state.add_memlet_path(tasklet, mx, dst_node, src_conn='out', memlet=dace.Memlet(data=r, subset='i'))

def partitionState(state, A, res_low, res_high):
    src_A = state.add_read(A)

    dst_node_low = state.add_write(res_low)
    dst_node_high = state.add_write(res_high)

    me_low, mx_low = state.add_map ('low', dict(i = '0:dSize/2'))
    me_high, mx_high = state.add_map('high', dict(j = 'dSize/2: dSize'))

    tasklet_low = state.add_tasklet('write_Low', {'_in'}, {'out_low'}, 'out_low = _in')
    tasklet_high = state.add_tasklet('write_High', {'_in'}, {'out_high'}, 'outhigh = _in')

    state.add_memlet_path(src_A, me_low, tasklet_low, dst_conn = '_in', memlet=dace.Memlet(data=A, subset='i'))
    state.add_memlet_path(src_A, me_high, tasklet_high, dst_conn='_in', memlet=dace.Memlet(data=A, subset='j'))

    state.add_memlet_path(tasklet_low, mx_low, dst_node_low, src_conn='out_low', memlet=dace.Memlet(data = res_low, subset='i'))
    state.add_memlet_path(tasklet_high, mx_high, dst_node_high, src_conn='out_high', memlet=dace.Memlet(data=res_high, subset='j'))

multiply_state = sdfg.add_state
hoistedState(multiply_state, 'A', 'B', 'Res')

partition_State = sdfg.add_state
partitionState(partition_State, 'Res', 'A', 'B')

block_sdfg = dace.SDFG('blockwise_Reduction')
bw_reduction_state = block_sdfg.add_state

main_state = sdfg.add_state


bme, bmx = bw_reduction_state.add_map('blockwise_map', dict(bId= '0:numBlocks'))

eme, emx = bw_reduction_state.add_map('elementwise_map', dict(i='0:blockSizes[bId]'))



sdfg(A=a, B=b, Res=res, dSize=size)
#sdfg.validate()
print("Ended")