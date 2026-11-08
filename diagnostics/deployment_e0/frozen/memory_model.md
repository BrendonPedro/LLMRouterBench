# E0 analytic memory model (prediction; F decides by measurement)

Qwen3-32B exact parameter counts: embedding 777,912,320; per layer 487,598,336; LM head 777,912,320; total 32,762,123,264 = 65,524,246,528 bytes bf16 (index total_size 65,524,246,528).

Context L_F = 17920 tokens, prefill chunk 2048, KV bf16 (4,096 B per token per layer). Assumed CUDA context 400 MiB + allocator slack 256 MiB.

| (k, j) | edge layers | weights MiB | KV MiB | activations MiB | fixed MiB | total MiB | fits 12 (11776) | fits 16 (15872) | fits 24 (24064) | server layers | server total MiB |
|---|---|---|---|---|---|---|---|---|---|---|---|
| (1, 1) | 2 | 4,828 | 140 | 753 | 656 | 6,377 | yes | yes | yes | 62 | 63,391 |
| (2, 2) | 4 | 6,688 | 280 | 753 | 656 | 8,377 | yes | yes | yes | 60 | 61,391 |
| (3, 3) | 6 | 8,548 | 420 | 753 | 656 | 10,377 | yes | yes | yes | 58 | 59,391 |
| (4, 4) | 8 | 10,408 | 560 | 753 | 656 | 12,377 | no | yes | yes | 56 | 57,391 |
| (5, 5) | 10 | 12,268 | 700 | 753 | 656 | 14,377 | no | yes | yes | 54 | 55,391 |
| (6, 6) | 12 | 14,128 | 840 | 753 | 656 | 16,377 | no | no | yes | 52 | 53,391 |
| (7, 7) | 14 | 15,988 | 980 | 753 | 656 | 18,377 | no | no | yes | 50 | 51,391 |
| (8, 8) | 16 | 17,848 | 1,120 | 753 | 656 | 20,377 | no | no | yes | 48 | 49,391 |
| (9, 9) | 18 | 19,708 | 1,260 | 753 | 656 | 22,377 | no | no | yes | 46 | 47,391 |
| (10, 10) | 20 | 21,568 | 1,400 | 753 | 656 | 24,377 | no | no | no | 44 | 45,391 |
| (11, 11) | 22 | 23,428 | 1,540 | 753 | 656 | 26,377 | no | no | no | 42 | 43,391 |

Predicted U-max(M): 12 GB -> (3, 3), 16 GB -> (5, 5), 24 GB -> (9, 9)

Edge-resident competitors (vLLM; weights + KV at L_F + assumed 1.5 GiB engine overhead):

| model | KV | predicted MiB | fits 12 | fits 16 | fits 24 |
|---|---|---|---|---|---|
| Qwen3-8B-AWQ | bf16/fp16 | 9,873 | yes | yes | yes |
| Qwen3-14B-AWQ | bf16/fp16 | 13,873 | no | yes | yes |
| Qwen3-32B-AWQ | bf16/fp16 | 24,422 | no | no | no |
| Qwen3-32B-AWQ | fp8 | 22,182 | no | no | yes |
| Qwen3-8B | bf16/fp16 | 19,696 | no | no | yes |

