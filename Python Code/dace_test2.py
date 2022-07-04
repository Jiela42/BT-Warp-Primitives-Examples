import cupy
import numpy as np
import dace
from dace.sdfg import nodes
from dace.transformation.auto import auto_optimize as auto
from dace.transformation.dataflow import MapExpansion


N = dace.symbol('N')
sz = 100000000


@dace.program
def hadamard_product_python(A: dace.float64[N], B: dace.float64[N]):
    return A * B


@dace.program
def add_reduce_python(A: dace.float64[N]):
    return np.add.reduce(A)


@dace.program
def inner_product_python(A: dace.float64[N], B: dace.float64[N]):
    return np.add.reduce(A * B)


def auto_gpu(prog: dace.parser.DaceProgram) -> dace.SDFG:
    sdfg = prog.to_sdfg(simplify=True)
    for _, arr in sdfg.arrays.items():
        if not arr.transient:
            arr.storage = dace.StorageType.GPU_Global
    auto.auto_optimize(sdfg, dace.DeviceType.GPU)
    return sdfg


def hadamard_product_sdfg():
    sdfg = dace.SDFG('hadamard_product_sdfg')
    sdfg.add_array('A', shape=[N], dtype=dace.float64, storage=dace.StorageType.GPU_Global)
    sdfg.add_array('B', shape=[N], dtype=dace.float64, storage=dace.StorageType.GPU_Global)
    sdfg.add_array('__return', shape=[N], dtype=dace.float64, storage=dace.StorageType.GPU_Global)

    state = sdfg.add_state()

    tasklet_code = f"""
    for (int id = i * 256 + j; id < N; id += blockDim.x * gridDim.x) {{
        out[id] = in1[id] * in2[id];
    }}
    """

    tasklet, me, mx = state.add_mapped_tasklet(
        name='callingKernel',
        map_ranges={'i': '0:min(int_ceil(N, 256), 2048)', 'j': '0:256'},
        inputs={'in1': dace.Memlet('A[0:N]'), 'in2': dace.Memlet('B[0:N]')},
        outputs={'out': dace.Memlet('__return[0:N]')},
        code=tasklet_code,
        language=dace.dtypes.Language.CPP,
        external_edges=True
    )
    out_conn = {'out': dace.pointer(dace.float64)}
    tasklet.out_connectors = out_conn
    me.map.schedule = dace.dtypes.ScheduleType.GPU_Device

    sdfg.apply_transformations_repeated(MapExpansion)
    for n in state.nodes():
        if isinstance(n, nodes.MapEntry) and "j" in n.map.params:
            n.map.schedule = dace.dtypes.ScheduleType.GPU_ThreadBlock

    return sdfg


def add_reduce_sdfg():
    sdfg = dace.SDFG('add_reduce_sdfg')
    sdfg.add_array('A', shape=[N], dtype=dace.float64, storage=dace.StorageType.GPU_Global)
    sdfg.add_array('__return', shape=[1], dtype=dace.float64, storage=dace.StorageType.GPU_Global)

    state = sdfg.add_state()

    tasklet_code = f"""
    if (i + j == 0) {{
        out[0] = double(0);
    }}
    double sum = double(0);
    for (int id = i * 256 + j; id < N; id += blockDim.x * gridDim.x) {{
        sum += in1[id];
    }}
    for (int offset = warpSize/2; offset > 0; offset /= 2) {{
        sum += __shfl_down_sync(0xFFFFFFFF, sum, offset);
    }}
    if (j % warpSize == 0) {{
        atomicAdd(out, sum);
    }}
    """

    tasklet, me, mx = state.add_mapped_tasklet(
        name='callingKernel',
        map_ranges={'i': '0:min(int_ceil(N, 256), 2048)', 'j': '0:256'},
        inputs={'in1': dace.Memlet('A[0:N]')},
        outputs={'out': dace.Memlet('__return[0]')},
        code=tasklet_code,
        language=dace.dtypes.Language.CPP,
        external_edges=True
    )
    out_conn = {'out': dace.pointer(dace.float64)}
    tasklet.out_connectors = out_conn
    me.map.schedule = dace.dtypes.ScheduleType.GPU_Device

    sdfg.apply_transformations_repeated(MapExpansion)
    for n in state.nodes():
        if isinstance(n, nodes.MapEntry) and "j" in n.map.params:
            n.map.schedule = dace.dtypes.ScheduleType.GPU_ThreadBlock

    return sdfg


def inner_produce_sdfg():
    sdfg = dace.SDFG('inner_product_sdfg')
    sdfg.add_array('A', shape=[N], dtype=dace.float64, storage=dace.StorageType.GPU_Global)
    sdfg.add_array('B', shape=[N], dtype=dace.float64, storage=dace.StorageType.GPU_Global)
    sdfg.add_array('__return', shape=[1], dtype=dace.float64, storage=dace.StorageType.GPU_Global)

    state = sdfg.add_state()

    tasklet_code = f"""
    if (i + j == 0) {{
        out[0] = double(0);
    }}
    double sum = double(0);
    for (int id = i * 256 + j; id < N; id += blockDim.x * gridDim.x) {{
        sum += in1[id] * in2[id];
    }}
    for (int offset = warpSize/2; offset > 0; offset /= 2) {{
        sum += __shfl_down_sync(0xFFFFFFFF, sum, offset);
    }}
    if (j % warpSize == 0) {{
        atomicAdd(out, sum);
    }}
    """

    tasklet, me, mx = state.add_mapped_tasklet(
        name='callingKernel',
        map_ranges={'i': '0:min(int_ceil(N, 256), 2048)', 'j': '0:256'},
        inputs={'in1': dace.Memlet('A[0:N]'), 'in2': dace.Memlet('B[0:N]')},
        outputs={'out': dace.Memlet('__return[0]')},
        code=tasklet_code,
        language=dace.dtypes.Language.CPP,
        external_edges=True
    )
    out_conn = {'out': dace.pointer(dace.float64)}
    tasklet.out_connectors = out_conn
    me.map.schedule = dace.dtypes.ScheduleType.GPU_Device

    sdfg.apply_transformations_repeated(MapExpansion)
    for n in state.nodes():
        if isinstance(n, nodes.MapEntry) and "j" in n.map.params:
            n.map.schedule = dace.dtypes.ScheduleType.GPU_ThreadBlock

    return sdfg


if __name__ == "__main__":

    hprod_py = auto_gpu(hadamard_product_python)
    ared_py = auto_gpu(add_reduce_python)
    iprod_py = auto_gpu(inner_product_python)
    hprod_sd = hadamard_product_sdfg()
    ared_sd = add_reduce_sdfg()
    iprod_sd = inner_produce_sdfg()

    rng = np.random.default_rng(42)
    A = rng.random((sz,))
    gA = cupy.asarray(A)
    B = rng.random((sz,))
    gB = cupy.asarray(B)

    print()
    print("Hadamard product - Python")
    gC_py = hprod_py(A=gA, B=gB, N=sz)
    print()
    print("Hadamard product - SDFG with CUDA code")
    gC_sd = hprod_sd(A=gA, B=gB, N=sz)
    assert(np.allclose(gC_py, gC_sd))

    print()
    print("Sum reduction - Python (CUB Device expansion)")
    red_py = ared_py(A=gC_py, N=sz)
    print()
    print("Sum reduction - SDFG with CUDA code")
    red_sd = ared_sd(A=gC_sd, N=sz)
    assert(np.allclose(red_py, red_sd))

    print()
    print("Inner product - Python")
    res_py = iprod_py(A=gA, B=gB, N=sz)
    print()
    print("Inner product - SDFG with CUDA code")
    res_sd = iprod_sd(A=gA, B=gB, N=sz)
    assert(np.allclose(res_py, res_sd))
