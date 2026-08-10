# Benchmark report

> **Integrity notes**
> - 13 run(s) had run-to-run spread above 10%, which usually means the machine was not idle

## Devices

**Devices under test**

| device | cpu | arch | cores | RAM GB | DRAM peak GB/s | runs |
|---|---|---|---|---|---|---|
| laptop | 12th Gen Intel(R) Core(TM) i7-12700H | AMD64 | 14 | 34.01 | 53.90 | 75 |

## Throughput

**LLM decode and prefill throughput**

| device | model | quant | thr | MB | prefill t/s | decode t/s | GB/s | % peak | W | C |
|---|---|---|---|---|---|---|---|---|---|---|
| laptop | qwen0.5b-q2k | Q2_K | 8 | 332.7 | 443.6 | 107.3 | 35.70 | 66.23 | - | - |
| laptop | qwen0.5b-q4km | Q4_K_M | 1 | 391.9 | 55.04 | 21.67 | 8.49 | 15.75 | - | - |
| laptop | qwen0.5b-q4km | Q4_K_M | 4 | 391.9 | 192.8 | 69.39 | 27.19 | 50.45 | - | - |
| laptop | qwen0.5b-q4km | Q4_K_M | 8 | 391.9 | 92.27 | 31.77 | 12.45 | 23.10 | - | - |
| laptop | qwen0.5b-q4km | Q4_K_M | 8 | 391.9 | 166.0 | 60.63 | 23.76 | 44.08 | - | - |
| laptop | qwen0.5b-q4km | Q4_K_M | 8 | 391.9 | 230.8 | 88.47 | 34.67 | 64.32 | - | - |
| laptop | qwen0.5b-q4km | Q4_K_M | 8 | 391.9 | 194.7 | 88.68 | 34.75 | 64.47 | - | - |
| laptop | qwen0.5b-q4km | Q4_K_M | 8 | 391.9 | 218.7 | 91.82 | 35.98 | 66.75 | - | - |
| laptop | qwen0.5b-q4km | Q4_K_M | 8 | 391.9 | 229.5 | 86.64 | 33.95 | 62.99 | - | - |
| laptop | qwen0.5b-q4km | Q4_K_M | 8 | 391.9 | 244.0 | 87.93 | 34.46 | 63.93 | - | - |
| laptop | qwen0.5b-q4km | Q4_K_M | 8 | 391.9 | 237.2 | 90.21 | 35.35 | 65.58 | - | - |
| laptop | qwen0.5b-q4km | Q4_K_M | 8 | 391.9 | 209.7 | 77.49 | 30.36 | 56.34 | - | - |
| laptop | qwen0.5b-q4km | Q4_K_M | 8 | 391.9 | 233.5 | 78.21 | 30.65 | 56.86 | - | - |
| laptop | qwen0.5b-q4km | Q4_K_M | 8 | 391.9 | 217.3 | 91.02 | 35.67 | 66.17 | - | - |
| laptop | qwen0.5b-q4km | Q4_K_M | 14 | 391.9 | 221.5 | 86.83 | 34.02 | 63.12 | - | - |
| laptop | qwen0.5b-q4km | Q4_K_M | 20 | 391.9 | 236.9 | 56.38 | 22.09 | 40.99 | - | - |
| laptop | fp16src-q2k | Q2_K | 8 | 409.2 | 414.8 | 100.3 | 41.05 | 76.15 | - | - |
| laptop | fp16src-iq4xs | IQ4_XS | 8 | 422.1 | 422.4 | 108.3 | 45.71 | 84.81 | - | - |
| laptop | fp16src-q40 | Q4_0 | 8 | 422.8 | 585.2 | 117.6 | 49.71 | 92.23 | - | - |
| laptop | fp16src-iq4nl | IQ4_NL | 8 | 424.9 | 536.1 | 110.8 | 47.06 | 87.32 | - | - |
| laptop | fp16src-q3km | Q3_K_M | 8 | 426.1 | 505.1 | 107.8 | 45.92 | 85.20 | - | - |
| laptop | fp16src-q4km | Q4_K_M | 8 | 485.5 | 242.8 | 90.14 | 43.76 | 81.18 | - | - |
| laptop | qwen0.5b-q8 | Q8_0 | 8 | 525.1 | 233.7 | 82.12 | 43.12 | 80.01 | - | - |
| laptop | fp16src-q6k | Q6_K | 8 | 644.4 | 282.9 | 81.99 | 52.84 | 98.03 | - | - |
| laptop | fp16src-q80 | Q8_0 | 8 | 669.8 | 277.1 | 78.79 | 52.77 | 97.90 | - | - |
| laptop | llama1b-q4km | Q4_K_M | 1 | 799.9 | 36.87 | 12.66 | 10.13 | 18.79 | - | - |
| laptop | llama1b-q4km | Q4_K_M | 4 | 799.9 | 133.2 | 38.52 | 30.81 | 57.16 | - | - |
| laptop | llama1b-q4km | Q4_K_M | 8 | 799.9 | 194.0 | 49.72 | 39.77 | 73.78 | - | - |
| laptop | llama1b-q4km | Q4_K_M | 14 | 799.9 | 227.5 | 48.83 | 39.06 | 72.46 | - | - |
| laptop | llama1b-q4km | Q4_K_M | 20 | 799.9 | 239.3 | 40.71 | 32.56 | 60.41 | - | - |
| laptop | qwen1.5b-q4km | Q4_K_M | 1 | 980.1 | 27.16 | 10.70 | 10.49 | 19.46 | - | - |
| laptop | qwen1.5b-q4km | Q4_K_M | 4 | 980.1 | 100.7 | 33.08 | 32.42 | 60.16 | - | - |
| laptop | qwen1.5b-q4km | Q4_K_M | 8 | 980.1 | 134.3 | 35.28 | 34.58 | 64.15 | - | - |
| laptop | qwen1.5b-q4km | Q4_K_M | 8 | 980.1 | 141.5 | 39.13 | 38.35 | 71.15 | - | - |
| laptop | qwen1.5b-q4km | Q4_K_M | 8 | 980.1 | 53.72 | 17.17 | 16.83 | 31.22 | - | - |
| laptop | qwen1.5b-q4km | Q4_K_M | 8 | 980.1 | 100.8 | 29.78 | 29.19 | 54.15 | - | - |
| laptop | qwen1.5b-q4km | Q4_K_M | 14 | 980.1 | 165.8 | 38.74 | 37.97 | 70.45 | - | - |
| laptop | qwen1.5b-q4km | Q4_K_M | 20 | 980.1 | 166.1 | 32.08 | 31.44 | 58.33 | - | - |
| laptop | qwen3b-q4km | Q4_K_M | 1 | 1,924 | 12.81 | 5.60 | 10.77 | 19.99 | - | - |
| laptop | qwen3b-q4km | Q4_K_M | 4 | 1,924 | 48.16 | 17.01 | 32.72 | 60.71 | - | - |
| laptop | qwen3b-q4km | Q4_K_M | 8 | 1,924 | 69.01 | 20.62 | 39.67 | 73.59 | - | - |
| laptop | qwen3b-q4km | Q4_K_M | 14 | 1,924 | 80.16 | 20.49 | 39.43 | 73.15 | - | - |
| laptop | qwen3b-q4km | Q4_K_M | 20 | 1,924 | 84.33 | 18.43 | 35.46 | 65.79 | - | - |
| laptop | qwen7b-q4km | Q4_K_M | 1 | 4,677 | 5.56 | 2.31 | 10.81 | 20.06 | - | - |
| laptop | qwen7b-q4km | Q4_K_M | 4 | 4,677 | 20.98 | 7.75 | 36.26 | 67.26 | - | - |
| laptop | qwen7b-q4km | Q4_K_M | 8 | 4,677 | 29.42 | 9.60 | 44.90 | 83.30 | - | - |
| laptop | qwen7b-q4km | Q4_K_M | 14 | 4,677 | 35.02 | 10.40 | 48.62 | 90.21 | - | - |
| laptop | qwen7b-q4km | Q4_K_M | 20 | 4,677 | 37.45 | 10.07 | 47.10 | 87.38 | - | - |

## Decode roofline

**Decode roofline: tokens per second against inverse model size**

| device | points | effective GB/s | R^2 | DRAM peak GB/s | % of peak |
|---|---|---|---|---|---|
| laptop | 15 | 43.11 | 0.89 | 53.90 | 79.99 |

## Thread scaling

**Thread scaling, relative to the lowest thread count measured**

| device | model | threads | prefill t/s | decode t/s | prefill speedup | decode speedup |
|---|---|---|---|---|---|---|
| laptop | llama1b-q4km | 1 | 36.87 | 12.66 | 1.00 | 1.00 |
| laptop | llama1b-q4km | 4 | 133.2 | 38.52 | 3.61 | 3.04 |
| laptop | llama1b-q4km | 8 | 194.0 | 49.72 | 5.26 | 3.93 |
| laptop | llama1b-q4km | 14 | 227.5 | 48.83 | 6.17 | 3.86 |
| laptop | llama1b-q4km | 20 | 239.3 | 40.71 | 6.49 | 3.22 |
| laptop | qwen0.5b-q4km | 1 | 55.04 | 21.67 | 1.00 | 1.00 |
| laptop | qwen0.5b-q4km | 4 | 192.8 | 69.39 | 3.50 | 3.20 |
| laptop | qwen0.5b-q4km | 8 | 192.9 | 45.80 | 3.51 | 2.11 |
| laptop | qwen0.5b-q4km | 8 | 174.6 | 86.39 | 3.17 | 3.99 |
| laptop | qwen0.5b-q4km | 8 | 92.27 | 31.77 | 1.68 | 1.47 |
| laptop | qwen0.5b-q4km | 8 | 225.0 | 79.98 | 4.09 | 3.69 |
| laptop | qwen0.5b-q4km | 8 | 166.0 | 60.63 | 3.02 | 2.80 |
| laptop | qwen0.5b-q4km | 8 | 230.8 | 88.47 | 4.19 | 4.08 |
| laptop | qwen0.5b-q4km | 8 | 194.7 | 88.68 | 3.54 | 4.09 |
| laptop | qwen0.5b-q4km | 8 | 218.7 | 91.82 | 3.97 | 4.24 |
| laptop | qwen0.5b-q4km | 8 | 229.5 | 86.64 | 4.17 | 4.00 |
| laptop | qwen0.5b-q4km | 8 | 228.6 | 68.55 | 4.15 | 3.16 |
| laptop | qwen0.5b-q4km | 8 | 244.0 | 87.93 | 4.43 | 4.06 |
| laptop | qwen0.5b-q4km | 8 | 237.2 | 90.21 | 4.31 | 4.16 |
| laptop | qwen0.5b-q4km | 8 | 209.7 | 77.49 | 3.81 | 3.58 |
| laptop | qwen0.5b-q4km | 8 | 233.5 | 78.21 | 4.24 | 3.61 |
| laptop | qwen0.5b-q4km | 8 | 217.3 | 91.02 | 3.95 | 4.20 |
| laptop | qwen0.5b-q4km | 14 | 221.5 | 86.83 | 4.02 | 4.01 |
| laptop | qwen0.5b-q4km | 20 | 236.9 | 56.38 | 4.30 | 2.60 |
| laptop | qwen1.5b-q4km | 1 | 27.16 | 10.70 | 1.00 | 1.00 |
| laptop | qwen1.5b-q4km | 4 | 100.7 | 33.08 | 3.71 | 3.09 |
| laptop | qwen1.5b-q4km | 8 | 134.3 | 35.28 | 4.95 | 3.30 |
| laptop | qwen1.5b-q4km | 8 | 141.5 | 39.13 | 5.21 | 3.66 |
| laptop | qwen1.5b-q4km | 8 | 143.2 | 37.59 | 5.27 | 3.51 |
| laptop | qwen1.5b-q4km | 8 | 119.6 | 24.09 | 4.40 | 2.25 |
| laptop | qwen1.5b-q4km | 8 | 141.2 | 33.33 | 5.20 | 3.12 |
| laptop | qwen1.5b-q4km | 8 | 53.72 | 17.17 | 1.98 | 1.60 |
| laptop | qwen1.5b-q4km | 8 | 100.8 | 29.78 | 3.71 | 2.78 |
| laptop | qwen1.5b-q4km | 8 | 128.3 | 38.74 | 4.72 | 3.62 |
| laptop | qwen1.5b-q4km | 14 | 165.8 | 38.74 | 6.11 | 3.62 |
| laptop | qwen1.5b-q4km | 20 | 166.1 | 32.08 | 6.12 | 3.00 |
| laptop | qwen3b-q4km | 1 | 12.81 | 5.60 | 1.00 | 1.00 |
| laptop | qwen3b-q4km | 4 | 48.16 | 17.01 | 3.76 | 3.04 |
| laptop | qwen3b-q4km | 8 | 63.40 | 21.38 | 4.95 | 3.82 |
| laptop | qwen3b-q4km | 8 | 67.57 | 18.38 | 5.27 | 3.28 |
| laptop | qwen3b-q4km | 8 | 69.01 | 20.62 | 5.39 | 3.68 |
| laptop | qwen3b-q4km | 8 | 58.35 | 13.48 | 4.55 | 2.41 |
| laptop | qwen3b-q4km | 8 | 67.43 | 19.61 | 5.26 | 3.50 |
| laptop | qwen3b-q4km | 14 | 80.16 | 20.49 | 6.26 | 3.66 |
| laptop | qwen3b-q4km | 20 | 84.33 | 18.43 | 6.58 | 3.29 |
| laptop | qwen7b-q4km | 1 | 5.56 | 2.31 | 1.00 | 1.00 |
| laptop | qwen7b-q4km | 4 | 20.98 | 7.75 | 3.77 | 3.35 |
| laptop | qwen7b-q4km | 8 | 29.42 | 9.60 | 5.29 | 4.15 |
| laptop | qwen7b-q4km | 14 | 35.02 | 10.40 | 6.29 | 4.50 |
| laptop | qwen7b-q4km | 20 | 37.45 | 10.07 | 6.73 | 4.36 |

## Quantisation

**Quantisation frontier**

| device | model | quant | MB | decode t/s | GB/s | mJ/token |
|---|---|---|---|---|---|---|
| laptop | qwen0.5b-q2k | Q2_K | 332.7 | 107.3 | 35.70 | - |
| laptop | qwen0.5b-q4km | Q4_K_M | 391.9 | 91.82 | 35.98 | - |
| laptop | fp16src-q2k | Q2_K | 409.2 | 100.3 | 41.05 | - |
| laptop | fp16src-iq4xs | IQ4_XS | 422.1 | 108.3 | 45.71 | - |
| laptop | fp16src-q40 | Q4_0 | 422.8 | 117.6 | 49.71 | - |
| laptop | fp16src-iq4nl | IQ4_NL | 424.9 | 110.8 | 47.06 | - |
| laptop | fp16src-q3km | Q3_K_M | 426.1 | 107.8 | 45.92 | - |
| laptop | fp16src-q4km | Q4_K_M | 485.5 | 90.14 | 43.76 | - |
| laptop | qwen0.5b-q8 | Q8_0 | 525.1 | 82.12 | 43.12 | - |
| laptop | fp16src-q6k | Q6_K | 644.4 | 81.99 | 52.84 | - |
| laptop | fp16src-q80 | Q8_0 | 669.8 | 78.79 | 52.77 | - |
| laptop | llama1b-q4km | Q4_K_M | 799.9 | 49.72 | 39.77 | - |
| laptop | qwen1.5b-q4km | Q4_K_M | 980.1 | 39.13 | 38.35 | - |
| laptop | qwen3b-q4km | Q4_K_M | 1,924 | 20.62 | 39.67 | - |
| laptop | qwen7b-q4km | Q4_K_M | 4,677 | 10.40 | 48.62 | - |

## Latency

**Single-request latency**

| device | model | prompt tok | output tok | TTFT ms | e2e ms | decode t/s | implied prefill t/s |
|---|---|---|---|---|---|---|---|
| laptop | qwen0.5b-q4km | 64 | 64 | 366.6 | 1,096 | 86.39 | 174.6 |
| laptop | qwen0.5b-q4km | 256 | 64 | 1,138 | 1,926 | 79.98 | 225.0 |
| laptop | qwen0.5b-q4km | 1024 | 64 | 4,479 | 5,392 | 68.55 | 228.6 |
| laptop | qwen0.5b-q4km | 4096 | 64 | 21,230 | 22,662 | 45.80 | 192.9 |
| laptop | qwen1.5b-q4km | 64 | 64 | 498.8 | 2,117 | 38.74 | 128.3 |
| laptop | qwen1.5b-q4km | 256 | 64 | 1,787 | 3,468 | 37.59 | 143.2 |
| laptop | qwen1.5b-q4km | 1024 | 64 | 7,251 | 9,159 | 33.33 | 141.2 |
| laptop | qwen1.5b-q4km | 4096 | 64 | 34,248 | 36,882 | 24.09 | 119.6 |
| laptop | qwen3b-q4km | 64 | 64 | 1,010 | 3,960 | 21.38 | 63.40 |
| laptop | qwen3b-q4km | 256 | 64 | 3,796 | 6,971 | 19.61 | 67.43 |
| laptop | qwen3b-q4km | 1024 | 64 | 15,155 | 18,550 | 18.38 | 67.57 |
| laptop | qwen3b-q4km | 4096 | 64 | 70,200 | 74,873 | 13.48 | 58.35 |

## Vision

**ONNX Runtime inference latency**

| device | model | quant | shape | thr | mean ms | p50 | p95 | p99 | inf/s | W |
|---|---|---|---|---|---|---|---|---|---|---|
| laptop | mnv3s-fp32 | fp32 | 1x3x32x32 | 8 | 0.59 | 0.57 | 0.69 | 0.88 | 1,697 | - |
| laptop | mnv3s-fp32 | fp32 | 1x3x32x32 | 4 | 0.47 | 0.46 | 0.56 | 0.62 | 2,129 | - |
| laptop | mnv3s-fp32 | fp32 | 1x3x32x32 | 1 | 0.44 | 0.38 | 0.82 | 0.90 | 2,273 | - |
| laptop | mnv3s-int8 | int8 | 1x3x32x32 | 4 | 0.73 | 0.68 | 0.90 | 1.10 | 1,370 | - |
| laptop | mnv3s-int8 | int8 | 1x3x32x32 | 1 | 0.60 | 0.59 | 0.66 | 0.77 | 1,665 | - |
| laptop | mnv3s-int8 | int8 | 1x3x32x32 | 8 | 0.77 | 0.73 | 0.95 | 1.02 | 1,296 | - |
| laptop | resnet18-fp32 | fp32 | 1x3x32x32 | 8 | 0.50 | 0.50 | 0.58 | 0.63 | 1,983 | - |
| laptop | resnet18-fp32 | fp32 | 1x3x32x32 | 4 | 0.64 | 0.62 | 0.88 | 1.04 | 1,557 | - |
| laptop | resnet18-fp32 | fp32 | 1x3x32x32 | 1 | 1.63 | 1.57 | 2.24 | 3.01 | 613.3 | - |
| laptop | resnet18-int8 | int8 | 1x3x32x32 | 1 | 3.36 | 3.23 | 4.28 | 4.88 | 297.7 | - |
| laptop | resnet18-int8 | int8 | 1x3x32x32 | 8 | 2.63 | 2.54 | 3.09 | 3.62 | 379.4 | - |
| laptop | resnet18-int8 | int8 | 64x3x32x32 | 8 | 43.04 | 42.93 | 45.99 | 46.56 | 1,487 | - |
| laptop | resnet18-int8 | int8 | 16x3x32x32 | 8 | 12.76 | 12.71 | 14.14 | 14.67 | 1,254 | - |
| laptop | resnet18-int8 | int8 | 4x3x32x32 | 8 | 4.90 | 4.94 | 5.54 | 5.71 | 815.8 | - |
| laptop | resnet18-int8 | int8 | 1x3x32x32 | 4 | 2.78 | 2.72 | 3.32 | 3.52 | 358.8 | - |

## Figures

![fig_roofline](fig_roofline.png)
![fig_threads](fig_threads.png)
![fig_ttft](fig_ttft.png)
![fig_vision](fig_vision.png)
