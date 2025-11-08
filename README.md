# Face Expression (Emotion) Detection Robot

This project demonstrates training a face expression/emotion detection model (based on YOLOv8-nano) and deploying it to an Arduino UNO Q to control a robot (stationary pan/tilt mount).

## Training

### Installation

We will use a Docker image to train our object detection model, as it gives us the option to run it on most major operating systems. Note that there are two versions of the Dockerfile:

 * **Dockerfile.gpu** - Use this if you have access to an NVIDIA GPU that you want to use to speed up training
 * **Dockerfile.cpu** - Use this if you want to use just your CPU instead (slower training but will work across a variety of systems)

> **Note**: I will demonstrate everything using the GPU version. Switch to the CPU version (i.e. use `Dockerfile.cpu`) if you do not have a GPU.

Build the Docker image:

```sh
docker build -f Dockerfile.gpu -t fed-training-gpu .
```

Run it (Linux, macOS, or PowerShell in Windows):

```sh
docker run --rm -it --gpus all -p 8888:8888 -v "${PWD}/workspace:/workspace" fed-training-gpu
```

> **Note**: If you want to override the Jupyter Lab server and get an interactive terminal instead, simply add `/bin/bash` to the end of the above command. In the interactive terminal (for the GPU version only), you can run `nvidia-smi` to test the availability of CUDA (you should see CUDA version at `13.0` based on the installed application versions).

With the Jupyter Lab server running, you can click on the URL shown in the terminal (likely the one starting with `http://127.0.0.1:8888/lab?token=...`) to open a browser for Jupyter Lab.

