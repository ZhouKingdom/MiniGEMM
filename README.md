# MiniGEMM
> 手写CUDA矩阵乘法库，基于A100 Ampere架构，迭代实现**计算受限Tensor Core内核**。4096³ FP16可达170 TFLOP/s（cuBLAS的71%），1024³性能反超cuBLAS（103%）。所有性能在同GPU同进程下与cuBLAS对比，每轮优化由Nsight Compute分析驱动。


## 项目亮点
- 从朴素GEMM逐步优化：共享内存分块、寄存器分块、`cp.async`双缓冲、`mma.sync`/`ldmatrix`、共享内存重排（swizzle）消除bank冲突，将内核从访存受限转为计算受限
- 严谨迭代：单变量改动验证优化效果，性能不升反降的方案保留记录并回滚
- 库化实现：支持C ABI、Fortran绑定，可直接替换BLAS；附带SpMV、FFT卷积、归约扫描等算子，对照cuSPARSE/cuFFT/CUB
- 屋顶线分析、自校验测试、分块自动搜索

## 性能
4096³优化阶梯：
| 优化阶段 | 精度 | GFLOP/s | 相对cuBLAS |
|---|---|---|---|
| naive | FP32 | 2596 | 13.7% |
| 共享内存分块 | FP32 | 4032 | 21.3% |
| 寄存器分块 | FP32 | 15086 | 79.5% |
| cp.async双缓冲 | FP32 | 16224 | 85.5% |
| WMMA | FP16 | 27555 | 11.5% |
| mma.sync | FP16 | 42867 | 17.8% |
| mma.sync+ldmatrix | FP16 | 43157 | 17.9% |
| ✅ 最终最优 | FP16 | 170327 | 70.8% |
| cuBLAS | FP16 | 240534 | 100% |

不同规模FP16相对cuBLAS：
| 规模 | 占比 |
|---|---|
| 512³ | 81% |
| 768³ | 85% |
| 1024³ | 103% |
| 2048³ | 83% |
| 4096³ | 71% |
| 8192³ | 68% |

硬件：A100-SXM4-80GB（sm_80），CUDA 13.2，Linux

## 复现
```bash
cmake --preset default
ninja -C build
ctest --test-dir build -L gpu -LE slow
./build/benchmarks/bench_all gemm auto fp16 4096 4096 4096
```
