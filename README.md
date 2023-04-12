# What is GLaDOS?
GLaDOS is a 20B model tuned to provide an open-source experience _similar_ _to_ ChatGPT. 

This repo includes the model itself and a basic web server to chat with it.


## Motivation
Similar models exist but often utilize LLAMA which is only available under a noncommercial license. GLaDOS avoids this by utilizing EleutherAI's/togethercomputers apach 2.0 licensed base models and CC0 data.

Additionally, GLaDOS is designed to be run fully standalone so you don't need to worry about your information being collected by a third party.

## Quickstart

GLaDOS is designed to run with docker. Instructions for installing docker https://docs.docker.com/get-docker/

First start the redis server needed to cache conversations
```
bash start_redis.sh
```
Second, build and run the GLaDOS server image/container with

```
bash build_and_run.sh
```

Then, from inside this container run 
```
python src/run_server.py
```
or
```
accelerate launch --multi_gpu --mixed_precision=fp16 --num_processes=1 src/run_server.py
```


The first time the model runs it will download the base model, which is `togethercomputer/GPT-NeoXT-Chat-Base-20B`.

GLaDOS is fine-tuned on ShareGPT data. ShareGPT data is available under a CC0 (No rights reserved) license https://huggingface.co/datasets/RyokoAI/ShareGPT52K

If you want to leave the server running you can build the container inside tmux, or modify the docker file to run the server directly.

## License
Apache 2.0 License, see LICENSE.md

## Examples
Basic Code Generation (Emphasis on basic)
![code example](images/code_generation_example.png)

Summarization and follow up questions
![follow up questions](images/follow_up_questions.png)

Brainstorming
![brainstorming example](images/mystery.png)

## Resource Requirements
The current version of GLaDOS uses an FP16 model with ~20B parameters. This is runnable in just under 48GB of VRAM by modifying the generation options in run_server to use a beam width of 1. I am running this with two A6000's nvlinked together and so the default settings run on multiGPU.

It should be possible to use GPTQ to reduce the memory requirements to ~16GB so that the model can be run on consumer grade graphics cards.

## Misc QnA

Q : Is the model as good as ChatGPT?

A : No, GLaDOS is only trained with SFT (no RLHF) on a relatively small (~50k) examples and uses a base model that is trained with less data, and fewer parameters, than OpenAI's GPT4 or even the larger/later iteration of GPT3 models. OpenAI has far more data and resources that make it possible to create bots like ChatGPT.

Q : If your model is trained on ChatGPT responses why doesn't it think it is ChatGPT?

A : Data has been transformed and filtered to remove OpenAI/ChatGPT related prompts. I leave items that only talk about it being a language model, so it has some sense of what it is, but it will often hallucinate information about who created it.

Q : How does the model handle formatting?

A : GLaDOS uses a slight variation on github flavored markdown to create lists tables and code blocks. Extra tags are added by the webserver to prettify the code blocks and tweak other small things.


