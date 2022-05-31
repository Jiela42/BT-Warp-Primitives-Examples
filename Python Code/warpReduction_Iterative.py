
import numpy as np
import dace

 # Da goal: sum ai * bi

# we start with the size = 32 = warpSize
size = 1 << 5

a = np.ones((size), dtype=float)
b = np.ones((size), dtype=float)
res = np.zeros((size), dtype=float)

res_vec = np.multiply(a,b)
checkSum = np.sum(res_vec);

# print(checkSum)

# Size = dace.symbol('Size')

sdfg = dace.SDFG('reduction')

# sdfg.add_symbol('Size', stype=dace.int32)
sdfg.add_scalar('dSize', dtype=dace.int32)
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

def changeSize(state):
    sr = state.add_read('dSize')
    sw = state.add_write('dSize')
   # lastSize = 'dSize'.clone()

    tasklet= state.add_tasklet('half', {'a'}, {'b'}, 'b = a/2')    

    state.add_edge(sr, None, tasklet, 'a', memlet=dace.Memlet(data='dSize', subset = '0'))
    state.add_edge(tasklet, None, sw, 'b', memlet=dace.Memlet(data='dSize', subset= '0'))

def reductionStep(state, A, B, r):
    src_A = state.add_read(A)
    src_B = state.add_read(B)

    dst_node = state.add_write(r)

    me, mx = state.add_map('reduceMap', dict(i = '0:dSize'))
    tasklet = state.add_tasklet('reduce', {'in1', 'in2'},{'out'}, 'out = in1 + in2')

    state.add_memlet_path(src_A, me, tasklet, dst_conn='in1', memlet=dace.Memlet(data=A, subset='i'))
    state.add_memlet_path(src_B, me, tasklet, dst_conn='in2', memlet=dace.Memlet(data=B, subset = 'i'))
    state.add_memlet_path(tasklet, mx, dst_node, src_conn='out', memlet=dace.Memlet(data=r, subset='i'))


state0 = sdfg.add_state()
hoistedState(state0, 'A', 'B', 'Res')

# instead I might wanna use a fancy interstate edge that does some partitioning <3
state_Split = sdfg.add_state()
partitionState(state_Split, 'Res', 'A', 'B')

state_Half_Size = sdfg.add_state()
changeSize(state_Half_Size)

state_reductionStep = sdfg.add_state()
reductionStep(state_reductionStep, 'A', 'B', 'Res')

# end_state = sdfg.add_state()
# write_back(end_state, 'Res')


sdfg.add_edge(state0, state_Split, dace.InterstateEdge())
sdfg.add_edge(state_Split, state_Half_Size, dace.InterstateEdge())
sdfg.add_edge(state_Half_Size, state_reductionStep, dace.InterstateEdge('dSize' > 1))
# sdfg.add_edge(state_Half_Size, end_state, dace.InterstateEdge('dSize' == 1))

#def reductionState(state, A, B, r):

# print(res[0])

sdfg(A=a, B=b, Res=res, dSize=size)
#sdfg.validate()
print("Ended")