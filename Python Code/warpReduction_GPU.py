from math import floor
import numpy as np
import dace
from dace.transformation.interstate import GPUTransformSDFG
from dace.transformation.interstate import StateFusion
from dace.transformation.dataflow import MapExpansion

from dace.sdfg import nodes

# Da goal: sum ai * bi
# Specifically: put some of this into the gpu (just SOMETHING!!)

size = (1 << 10)
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

sdfg.add_transient('gA', [size], dace.float64, dace.StorageType.GPU_Global)
sdfg.add_transient('gB', [size], dace.float64, dace.StorageType.GPU_Global)
sdfg.add_transient('gRes', [size], dace.float64, dace.StorageType.GPU_Global)

floorPow = 1

while (floorPow < size):
    floorPow = floorPow << 1

if floorPow > size:
    floorPow = floorPow >> 1

overshoot = size - floorPow
size = floorPow



def KernelCall(state):

    tasklet_code = f"""
    /* since every thread adds up two numbers (in the first iteration)
    we need double the block-starting point to keep working on disjoint parts of the input */
    int id = j + i;
    int stepSize = Size/2;

    int iterations = 1;
    int logSize = Size;

    while(logSize > 1){{
        logSize /= 2;
        iterations++;
    }}

    // first iteration: 
    out[id] = (in1[id] * in2[id]) + (in1[id + stepSize] * in2[id + stepSize]);
    stepSize /= 2; 

    for(int i = 1; i < iterations; i++){{
        if (id - 2 * (blockDim.x * blockIdx.x) < stepSize){{
            out[id] += out[id + stepSize];
            stepSize /= 2;
        }}
        __syncthreads();
    }}
    """


    tasklet, me, mx = state.add_mapped_tasklet(
        name='callingKernel',
        map_ranges={'i': '0:Size/2:64', 'j': '0:64'},
        inputs={'in1': dace.Memlet('gA[0:Size]'), 'in2': dace.Memlet('gB[0:Size]')},
        outputs={'out': dace.Memlet('gRes[0:Size]')},
        code=tasklet_code,
        language=dace.dtypes.Language.CPP,
        external_edges=True
    )
    me.map.schedule = dace.dtypes.ScheduleType.GPU_Device
    
    for n in state.nodes():
        if isinstance(n, nodes.AccessNode):
            if n.data == 'gA':
                gA = n
            elif n.data == 'gB':
                gB = n
            elif n.data == 'gRes':
                gRes = n
    # tasklet = state.add_tasklet(name='callingKernel',
    #                             inputs={'in1', 'in2'},
    #                             outputs={'out'},
    #                             code=
    # ''' void* args[] = {&in1, &in2, &out, &Size};
    # cudaLaunchKernel((void*)naiveGlobalMem, dim3 (1), dim3(Size), args, 0, __dace_current_stream);''',
    #                             language=dace.Language.CPP)
    A = state.add_read('A')
    B = state.add_read('B')
    Res = state.add_write('Res')
    # gA = state.add_access('gA')
    # gB = state.add_access('gB')
    # gRes = state.add_access('gRes')

    # state.add_edge(gA, None, tasklet, 'in1', dace.Memlet('gA[0:Size]'))
    # state.add_edge(gB, None, tasklet, 'in2', dace.Memlet('gB[0:Size]'))
    # state.add_edge(tasklet, 'out', gRes, None, dace.Memlet('gRes[0:Size]'))

    state.add_nedge(A, gA, dace.Memlet(data='gA', subset='0:Size'))
    state.add_nedge(B, gB, dace.Memlet(data='gB', subset=' 0:Size'))
    state.add_nedge(gRes, Res, dace.Memlet(data='Res', subset='0:Size'))


callState = sdfg.add_state()
KernelCall(callState)

sdfg.apply_transformations_repeated(MapExpansion)

for n in callState.nodes():
    if isinstance(n, nodes.MapEntry) and "j" in n.map.params:
        n.map.schedule = dace.dtypes.ScheduleType.GPU_ThreadBlock
        

# sdfg.append_global_code('''
# __global__ void naiveGlobalMem(double * a, double * b, double* res, int höck){

    # /* since every thread adds up two numbers (in the first iteration)
    # we need double the block-starting point to keep working on disjoint parts of the input */
    # int id = threadIdx.x + 2*(blockIdx.x * blockDim.x);
    # int stepSize = höck/2;

    # int iterations = 1;
    # int logSize = höck;

    # while(logSize > 1){
    #     logSize /= 2;
    #     iterations++;
    # }

    # // first iteration: 
    # res[id] = (a[id] * b[id]) + (a[id + stepSize] * b[id + stepSize]);
    # stepSize /= 2; 

    # for(int i = 1; i < iterations; i++){
    #     if (id - 2 * (blockDim.x * blockIdx.x) < stepSize){
    #         res[id] += res[id + stepSize];
    #         stepSize /= 2;
    #     }
    #     __syncthreads();
    # }
# }
# ''')


sdfg(A=a, B=b, Res=res, Size=size, Overshoot=overshoot)

#sdfg.validate()
print(res)
print("Ended")
