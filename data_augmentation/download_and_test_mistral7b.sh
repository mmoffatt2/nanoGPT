#!/bin/bash

# This script will downlaod and test Mistral7B using llama-cpp-python

if [ -d ./models ]; then
  mkdir -p ./models
fi

if [ -f "./models/mistral-7b-instruct-v0.2.Q5_K_M.gguf" ]; then
  echo "./models/mistral-7b-instruct-v0.2.Q5_K_M.gguf file found, continuing"
else
  echo "./models/mistral-7b-instruct-v0.2.Q5_K_M.gguf file not found, downloading"
  wget -P ./models https://huggingface.co/TheBloke/Mistral-7B-Instruct-v0.2-GGUF/resolve/main/mistral-7b-instruct-v0.2.Q5_K_M.gguf
fi

python3 llama-cpp-python_example.py

