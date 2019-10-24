Launch docker

NV_GPU=0,1 nvidia-docker run -p 1234:8888 -p 1235:6006 -v /mnt/scratch:/mnt/scratch -v /mnt/datasets/public/:/mnt/datasets/public/ -v /mnt/home/boris:/mnt/home/boris -t -d --name fewshot_webly_$USER images.borgy.elementai.net/fewshot_webly:$USER

Enter docker

docker exec -i -t fewshot_webly_$USER /bin/bash
