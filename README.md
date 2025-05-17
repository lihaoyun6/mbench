## mbench
Simple CUDA matmul benchmark CLI tool

## Usage

```bash
usage: perf_test.py [-h] [-i INDEX] [-d] [-s] [-b] [-j]

CUDA matmul benchmark script

options:
  -h, --help            show this help message and exit
  -i INDEX, --index INDEX
                        Run on the specified device
  -d, --double          Run FP64 benchmarks
  -s, --single          Run FP32 benchmarks
  -b, --bfloat          Run BF16 benchmarks
  -j, --json            Output in json format
```
