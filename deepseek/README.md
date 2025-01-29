# Deepseek R1

The new [Deepseek R1](https://github.com/deepseek-ai/DeepSeek-R1) was released on January 20th, 2025 with 671b parameters. With a reported development costs of 5.5 million USD (see paper page 5) it gained a lot of interest in the coming days, affecting the stock prices of many tech companies in the following days.

## Performance of distilled models

Several smaller models were released to run on consumer hardware, since the full model still requires 404 GB VRAM to run with the full 671 billion parameters. I tried to run a few of them on my hardware to test the performance.

### 1.5b model based on qwen2 1.5B

The [ollama website](https://ollama.com/library/deepseek-r1:1.5b) states that the Q4_K_M requires 1.1GB. It has 29 layers, and `ollama ps` reports 1.1 to 2.0 GB used VRAM, more on GPUs.

| CPU/GPU        | `ollama ps` | memory_required | token/s |
|----------------|:-----------:|:---------------:|--------:|
| Jetson Nano    |      1.1 GB |         1.0 GiB |    3.13 |
| Raspberry Pi 4 |      1.6 GB |         1.5 GiB |    3.14 |
| i3-6100        |      1.1 GB |         1.0 GiB |   15.53 |
| 1060           |             |                 |         |
| 3060 Ti        |      2.0 GB |         1.9 GiB |  119.33 |
| 3070 Ti        |      2.0 GB |         1.9 GiB |  145.45 |

### 7b model based on qwen 7B

### 8b model based on llama 8B

### 14b model based on qwen 14B

### 32b model based on qwen 32B

### 70b model based on llama3 70b

### 671b model

I have no hardware to run a model that needs 404 GB when qunatized to 4bit.


## Documentation

Deepseek has some documentation in [their github profile](https://github.com/deepseek-ai/DeepSeek-R1/tree/main). This includes:

- [A paper describing the model](https://github.com/deepseek-ai/DeepSeek-R1/blob/main/DeepSeek_R1.pdf), 20 pages
- [A technical report](https://github.com/deepseek-ai/DeepSeek-V3/blob/main/DeepSeek_V3.pdf), 48 pages
