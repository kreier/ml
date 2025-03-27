# Compiling and running llama.cpp

The [instructions](https://github.com/ggerganov/llama.cpp/blob/master/docs/build.md) are available on the llama.cpp website, but in general miss the libcurls, which gets the error message

``` 
llama_load_model_from_hf: llama.cpp built without libcurl, downloading from Hugging Face not supported.
```

Therefore my procedure is:

``` sh
git clone https://github.com/ggml-org/llama.cpp
cd llama.cpp
cmake -B build -DLLAMA_CURL=ON
cmake --build build --config Release
```

Also, if I remember correctly you might need the libcurl4-openssl-dev package as well. Yep, [discussion from October 2024](https://github.com/ggml-org/llama.cpp/discussions/9835).

```
sudo apt install libcurl4-openssl-dev
```

## Run the first model

``` sh
# Load and run a small model:
llama-cli -hf bartowski/DeepSeek-R1-Distill-Qwen-1.5B-GGUF
./build/bin/llama-cli -hf bartowski/DeepSeek-R1-Distill-Qwen-7B-GGUF
```

It will download the `GGUF` file to your `~/.cache/llama.cpp/` folder.

After a while you have your input prompt, and you can say simple things like `Hi` or ask questions like `How many R's are in the word STRAWBERRY`. To exit you type `Ctrl-C`.

## Compile with CUDA support

### Toolkit

Have the[ CUDA Toolkit](https://developer.nvidia.com/cuda-toolkit) installed. In my case its a few commands:

``` sh
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2404/x86_64/cuda-ubuntu2404.pin
sudo mv cuda-ubuntu2404.pin /etc/apt/preferences.d/cuda-repository-pin-600
wget https://developer.download.nvidia.com/compute/cuda/12.8.0/local_installers/cuda-repo-ubuntu2404-12-8-local_12.8.0-570.86.10-1_amd64.deb
sudo dpkg -i cuda-repo-ubuntu2404-12-8-local_12.8.0-570.86.10-1_amd64.deb
sudo cp /var/cuda-repo-ubuntu2404-12-8-local/cuda-*-keyring.gpg /usr/share/keyrings/
sudo apt-get update
sudo apt-get -y install cuda-toolkit-12-8
```

### Driver

You need to

``` sh
sudo apt-get install -y nvidia-open
```

### Compile

``` sh
cmake -B build -DGGML_CUDA=ON -DLLAMA_CURL=ON
cmake --build build --config Release
```

### Run with CUDA support

You need to set the `-ngl` or `--n-gpu-layers` to 999, otherwise only the CPU will be utilized 

``` sh
./build/bin/llama-cli -ngl 999 -hf bartowski/DeepSeek-R1-Distill-Qwen-1.5B-GGUF
```

## Benchmarking with llama-bench

### CPU

``` sh
$ ./llama-bench -m .cache/llama.cpp/DeepSeek-R1-Distill-Qwen-1.5B-Q4_K_M.gguf
| model                    |     size | params | backend | threads |   test |           t/s |
| ------------------------ | -------: | -----: | ------- | ------: | -----: | ------------: |
| qwen2 1.5B Q4_K - Medium | 1.04 GiB | 1.78 B | CPU     |       4 |  pp512 |  58.53 ± 6.88 |
| qwen2 1.5B Q4_K - Medium | 1.04 GiB | 1.78 B | CPU     |       4 |  tg128 |  24.18 ± 3.41 |

build: 19d3c829 (4677)
```

| model                    |     size | params | backend | threads |   test |           t/s |
| ------------------------ | -------: | -----: | ------- | ------: | -----: | ------------: |
| qwen2 1.5B Q4_K - Medium | 1.04 GiB | 1.78 B | CPU     |       4 |  pp512 |  58.53 ± 6.88 |
| qwen2 1.5B Q4_K - Medium | 1.04 GiB | 1.78 B | CPU     |       4 |  tg128 |  24.18 ± 3.41 |

### GPU

``` sh
$ ./llama-bench -m /mnt/data/models/DeepSeek-R1-1.5B-Q4_K_M.gguf -ngl 999
ggml_cuda_init: GGML_CUDA_FORCE_MMQ:    no
ggml_cuda_init: GGML_CUDA_FORCE_CUBLAS: no
ggml_cuda_init: found 1 CUDA devices:
  Device 0: NVIDIA GeForce RTX 3070 Ti, compute capability 8.6, VMM: yes
| model                    |     size |  params | backend | ngl |   test |               t/s |
| ------------------------ | -------: | ------: | ------- | --: | -----: | ----------------: |
| qwen2 1.5B Q4_K - Medium | 1.04 GiB |  1.78 B | CUDA    | 999 |  pp512 | 10292.97 ± 584.16 |
| qwen2 1.5B Q4_K - Medium | 1.04 GiB |  1.78 B | CUDA    | 999 |  tg128 |     233.28 ± 2.37 |

build: 19d3c829 (4677)
```

| model                    |     size |  params | backend | ngl |   test |               t/s |
| ------------------------ | -------: | ------: | ------- | --: | -----: | ----------------: |
| qwen2 1.5B Q4_K - Medium | 1.04 GiB |  1.78 B | CUDA    | 999 |  pp512 | 10292.97 ± 584.16 |
| qwen2 1.5B Q4_K - Medium | 1.04 GiB |  1.78 B | CUDA    | 999 |  tg128 |     233.28 ± 2.37 |

The old instructions were: `./llama.cpp/build/bin/llama-bench -m .cache/llama.cpp/bartowski_DeepSeek-R1-Distill-Qwen-1.5B-GGUF_DeepSeek-R1-Distill-Qwen-1.5B-Q4_K_M.gguf -ngl 999`

### Example TinyLlama-1.1B-Chat

Prompt to start and download the 4Q_K_M model:

``` sh
mk@i3:~/llama.cpp$ ./build/bin/llama-cli -hf TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF:Q4_K_M
```

#### Result CPU

```
| model                  |       size | params | backend | threads |  test |            t/s |
| ---------------------- | ---------: | -----: | ------- | ------: | ----: | -------------: |
| llama 1B Q4_K - Medium | 636.18 MiB | 1.10 B | CPU     |       4 | pp512 | 102.88 ± 19.54 |
| llama 1B Q4_K - Medium | 636.18 MiB | 1.10 B | CPU     |       4 | tg128 |   40.96 ± 0.49 |

build: f125b8dc (4977)
```

#### Result GPU

```
mk@i3:~/llama.cpp$ ./build/bin/llama-bench -m
../.cache/llama.cpp/TheBloke_TinyLlama-1.1B-Chat-v1.0-GGUF_tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf
ggml_cuda_init: GGML_CUDA_FORCE_MMQ:    no
ggml_cuda_init: GGML_CUDA_FORCE_CUBLAS: no
ggml_cuda_init: found 1 CUDA devices:
  Device 0: NVIDIA GeForce RTX 3070 Ti, compute capability 8.6, VMM: yes
| model                  |       size | params | backend | ngl |   test |               t/s |
| ---------------------- | ---------: | -----: | ------- | --: | -----: | ----------------: |
| llama 1B Q4_K - Medium | 636.18 MiB | 1.10 B | CUDA    |  99 |  pp512 | 12830.34 ± 186.18 |
| llama 1B Q4_K - Medium | 636.18 MiB | 1.10 B | CUDA    |  99 |  tg128 |    325.35 ± 11.50 |

build: f125b8dc (4977)
```

#### Jetson Nano

On a Jetson Nano this model only gets 4.98 tg128 and 6.71 pp512.
