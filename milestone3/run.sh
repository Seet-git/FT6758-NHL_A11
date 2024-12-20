#!/bin/bash

echo "TODO: fill in the docker run command"

export WANDB_API_KEY=your_wandb_api_key
docker run -e WANDB_API_KEY=${WANDB_API_KEY} -p 5000:5000 ift6758/serving