# What is GLaDOS?

GLaDOS is a 20B model tuned to provide an open-source experience _similar_ _to_ ChatGPT.

This repo includes the model itself and a basic web server to run it.


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

If you want to leave the server running you can build the container inside tmux, or modify the docker file to run the server directly.

The first time the model runs it will download the base model, which is `togethercomputer/GPT-NeoXT-Chat-Base-20B`.

GLaDOS is fine-tuned on ShareGPT data. ShareGPT data is available under a CC0 (No rights reserved) license https://huggingface.co/datasets/RyokoAI/ShareGPT52K

## License
Apache 2.0 License, see LICENSE.md

