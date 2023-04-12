# What is GLaDOS?

GLaDOS is a model trained to create responses similar to ChatGPT.
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

The first time the model runs it will download the model to 