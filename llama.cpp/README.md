# Compiling and running llama.cpp

The [instructions](https://github.com/ggerganov/llama.cpp/blob/master/docs/build.md) are available on the llama.cpp website, but in general miss the libcurls, which gets the error message

```
llama_load_model_from_hf: llama.cpp built without libcurl, downloading from Hugging Face not supported.
```

Therefore my procedure is:

``` sh
git clone https://github.com/ggerganov/llama.cpp
cd llama.cpp
cmake -B build -DLLAMA_CURL=ON
cmake --build build --config Release
```

Also, if I remember correctly you might need the libcurl4-openssl-dev package as well.

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

### Benchmark

``` sh
llama-bench -hf bartowski/DeepSeek-R1-Distill-Qwen-1.5B-GGUF
```
