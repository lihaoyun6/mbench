import argparse

def main():
    parser = argparse.ArgumentParser(description="CUDA matmul benchmark script")
    parser.add_argument("-i", "--index", type=int, help="Run on the specified device", default=0)
    parser.add_argument("-d", "--double", action="store_true", help="Run FP64 benchmarks")
    parser.add_argument("-s", "--single", action="store_true", help="Run FP32 benchmarks")
    parser.add_argument("-b", "--bfloat", action="store_true", help="Run BF16 benchmarks")
    parser.add_argument("-j", "--json", action="store_true", help="Output in json format")
    args = parser.parse_args()
    
    def var_dict(*args):
        callers_local_vars = inspect.currentframe().f_back.f_locals.items()
        return dict([(name, val) for name, val in callers_local_vars if val is arg][0] 
                    for arg in args)
    
    def walltime(stmt, arg_dict, duration=3):
        return benchmark.Timer(stmt=stmt, globals=arg_dict).blocked_autorange(
            min_run_time=duration).median
    
    import sys
    import torch
    
    try:
        if args.index < 0 or args.index >= torch.cuda.device_count():
            raise ValueError(f"Invalid device index: {args.index}")
    except ValueError as e:
        print(f"Error: {e}")
        sys.exit(1)

    with torch.cuda.device(args.index):
        print('CUDA Version\t:', torch.version.cuda)
        print('Torch Version\t:', torch.__version__)
        print('Device Name\t:', torch.cuda.get_device_name(args.index))
        
    if args.bfloat:
        if not torch.cuda.is_bf16_supported(args.index):
            print(f"Error: This device does not support the BF16 precision!")
            sys.exit(1)
            
    import json
    from tabulate import tabulate
    import inspect
    from collections import defaultdict
    import pandas as pd
    from torch.utils import benchmark

    pd.options.display.precision = 3
    matmul_tflops = defaultdict(lambda: {})

    dtypes = [torch.float16]
    if args.bfloat:
        dtypes.append(torch.bfloat16)
    if args.single:
        dtypes = [torch.float32] + dtypes
    if args.double:
        dtypes = [torch.float64] + dtypes

    for n in [512, 2048, 8192, 9216]:
        for dtype in dtypes:
            print(f'Running\t\t: {dtype} ({n}, {n})', end="    \r")
            a = torch.randn(n, n, dtype=dtype).cuda(args.index)
            b = torch.randn(n, n, dtype=dtype).cuda(args.index)
            t = walltime('a @ b', var_dict(a, b))
            matmul_tflops[f'n={n}'][dtype] = 2 * n**3 / t / 1e12
            del a, b

    converted = {
        n: {str(dtype): value for dtype, value in inner.items()}
        for n, inner in matmul_tflops.items()
    }

    sys.stdout.write('\033[K' + '\n')
    sys.stdout.flush()

    if args.json:
        print(json.dumps(converted, indent=2))
    else:
        dtype_name_map = {
            "torch.float64": "FP64",
            "torch.float32": "FP32",
            "torch.float16": "FP16",
            "torch.bfloat16": "BF16"
        }
        dtype_order = list(dtype_name_map.keys())
        sizes = sorted(converted.keys(), key=lambda x: int(x.split('=')[1]))
        headers = [] + [n.split('=')[1] for n in sizes]

        rows = []
        for d in dtype_order:
            if any(d in converted[n] for n in sizes):
                row = [dtype_name_map[d]]
                for n in sizes:
                    value = converted[n].get(d, "")
                    row.append(value if isinstance(value, float) else "")
                rows.append(row)

        print(tabulate(rows, headers=headers, tablefmt="grid", floatfmt=".2f"))