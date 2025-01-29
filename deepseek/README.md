# Deepseek R1

The new [Deepseek R1](https://github.com/deepseek-ai/DeepSeek-R1) was released on January 20th, 2025 with 671b parameters. With a reported development costs of 5.5 million USD (see paper page 5) it gained a lot of interest in the coming days, affecting the stock prices of many tech companies in the following days.

## Performance of distilled models

Several smaller models were released to run on consumer hardware, since the full model still requires 404 GB VRAM to run with the full 671 billion parameters. I tried to run a few of them on my hardware to test the performance.

### 1.5b model based on qwen2 1.5B

The [ollama website](https://ollama.com/library/deepseek-r1:1.5b) states that the Q4_K_M requires 1.1GB. It has 29 layers, and `ollama ps` reports 1.1 to 2.0 GB used VRAM, more on GPUs. It was not possible to export all of them on a RTX 960 with 2 GB VRAM.

| CPU/GPU        | `ollama ps` | memory_required | token/s |
|----------------|:-----------:|:---------------:|--------:|
| Jetson Nano    |      1.1 GB |         1.0 GiB |    3.13 |
| Raspberry Pi 4 |      1.6 GB |         1.5 GiB |    3.14 |
| i3-6100        |      1.1 GB |         1.0 GiB |   15.53 |
| RTX 1060       |      2.0 GB |         1.9 GiB |   61.21 |
| GTX 3060 Ti    |      2.0 GB |         1.9 GiB |  119.33 |
| GTX 3070 Ti    |      2.0 GB |         1.9 GiB |  145.45 |

### 7b model based on qwen 7B

The website states that it requres 4.7 GB, running on a GPU it uses 6.0 GB. For the 29 layers are 5.6 GiB required. With 7.62 B parameters we get 83 tokens/s.

### 8b model based on llama 8B

The website states that it requres 4.9 GB, running on a GPU it uses 5.8 GB. For the 33 layers are 5.4 GiB required. With 8.03 B parameters we get 79 tokens/s.

### 14b model based on qwen 14B

The website staes 9.0 GB for the Q4_K_M model, while `ollama ps` states 10 to 16 GB. Splitting up less increases performance. The model has 49 layers.

| `ollama ps` | memory_required | token/s | offload layers | CPU/GPU | GB GPUs |
|:-----------:|:---------------:|--------:|:--------------:|---------|---------|
|       10 GB |         9.5 GiB |     2.1 |                | 100/0   | 0       |
|       10 GB |        10.0 GiB |     4.1 | 30             | 37/63   | 8       |
|       16 GB |        15.6 GiB |    10.8 | 13/12/12/12    | 0/100   | 8/6/6/6 |
|       15 GB |        14.1 GiB |    14.2 | 17/16/16       | 0/100   | 8/6/6   |

### 32b model based on qwen 32B

The website staes 20 GB for the Q4_K_M model, while `ollama ps` states 22 to 26 GB. It currently does not fit into 4 GPUs with combined 26 GByte VRAM. The model has 65 layers.

| `ollama ps` | memory_required | token/s | offload layers | CPU/GPU | GB GPUs |
|:-----------:|:---------------:|--------:|:--------------:|---------|---------|
|       22 GB |        20.9 GiB |    0.92 |                | 100     |         |
|       25 GB |        23.6 GiB |    2.34 | 21/14/15       | 20/80   | 8/6/6   |
|       26 GB |        25.1 GiB |    5.11 | 19/15/15/15    | 2/98    | 8/6/6/6 |

### 70b model based on llama3 70b

To be tested on the E5-2696v3.

### 671b model

I have no hardware to run a model that needs 404 GB when qunatized to 4bit.


## Documentation

Deepseek has some documentation in [their github profile](https://github.com/deepseek-ai/DeepSeek-R1/tree/main). This includes:

- [A paper describing the model](https://github.com/deepseek-ai/DeepSeek-R1/blob/main/DeepSeek_R1.pdf), 20 pages
- [A technical report](https://github.com/deepseek-ai/DeepSeek-V3/blob/main/DeepSeek_V3.pdf), 48 pages
