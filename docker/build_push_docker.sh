#!/bin/bash

docker build -f Dockerfile_cluster -t fewshot_webly .
docker tag fewshot_webly images.borgy.elementai.net/fewshot_webly:$USER
docker push images.borgy.elementai.net/fewshot_webly:$USER
chmod +x ../main.py
