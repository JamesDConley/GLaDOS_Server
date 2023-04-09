#!/bin/bash
cd docker
parentdir=$(dirname `pwd`)

# If you want to use a different port number set it here
PORT=5950

# Build Docker Image
docker build -t "glados_server" .

# Run the docker container, forwarding the given PORT, and mounting the main repository under /app.
docker run -it --gpus all --rm --network glados-net --name glados_server -p $PORT:$PORT -v $parentdir:/app --env PORT=$PORT glados_server /bin/bash