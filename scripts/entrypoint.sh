#!/bin/bash

# Start TensorBoard in background
tensorboard --logdir=/workspace/runs --host=0.0.0.0 --port=6006 &

# Start Jupyter Lab in foreground
jupyter lab --ip=0.0.0.0 --port=8888 --no-browser --allow-root
