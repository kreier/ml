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
