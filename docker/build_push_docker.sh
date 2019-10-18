#!/bin/bash

docker build -f Dockerfile_cluster -t fewshot_webly .
docker tag fewshot_webly images.borgy.elementai.lan/tensorflow/fewshot_webly
docker push images.borgy.elementai.lan/tensorflow/fewshot_webly
chmod +x ../main.py
