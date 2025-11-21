# Face Expression (Emotion) Detection Robot

This project demonstrates training a face expression/emotion detection model (based on YOLOv8-nano) and deploying it to an Arduino UNO Q to control a robot (stationary pan/tilt mount).

## Required hardware

 * Arduino UNO Q
 * [USB webcam](https://www.amazon.com/dp/B0FVSMXGCN)
 * USB-C hub ([this one worked for me](https://www.amazon.com/dp/B0BQLLB61B))
 * Micro servos
 * LED ring
 * [3D printed pan/tilt mount](mechanical/pan-tilt-mount.zip) (note that this was custom designed for the webcam listed above)

## Install Jupyter Lab (in Docker image)

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
docker run --rm -it --gpus all --shm-size=2g -p 8888:8888 -p 6006:6006 -v "${PWD}/workspace:/workspace" fed-training-gpu
```

> **Note**: If you want to override the Jupyter Lab server and get an interactive terminal instead, simply add `--entrypoint /bin/bash` to the above command (before `fed-training-gpu`). In the interactive terminal (for the GPU version only), you can run `nvidia-smi` to test the availability of CUDA (you should see CUDA version at `13.0` based on the installed application versions).

With the Jupyter Lab server running, you can click on the URL shown in the terminal (likely the one starting with `http://127.0.0.1:8888/lab?token=...`) to open a browser for Jupyter Lab.

To connect to TensorBoard, navigate to [127.0.0.1:6006](127.0.0.1:6006).

## Training the model

In Jupyter Lab, open *workspace/notebooks/training.ipynb*. Follow the instructions as you run through each cell (*shift+enter* to run the cell). This should ultimately produce a trained model at *workspace/models/model.onnx*.

Copy *workspace/models/model.onnx* to *firmware/face-detection-robot/model.onnx*.

## Run the robot

Now, we'll copy the firmware project to the UNO Q and run it.

Connect a USB webcam through a USB hub to your UNO Q. Apply power to the hub such that it also powers the UNO Q. Once the UNO Q is powered, run [App Lab](https://docs.arduino.cc/software/app-lab/tutorials/getting-started/) at least once to ensure it has the latest firmware. Use App Lab to get the IP address of your UNO Q.

Open a terminal (to be used as your SSH connection to the UNO Q). Enter the following (replace <UNO_Q_IP_ADDRESS> with the IP address of your UNO Q board):

```sh
ssh arduino@<UNO_Q_IP_ADDRESS>
```

Enter yes if asked to accept the SSH key fingerprint. Enter your UNO Q password.

In that terminal, create a new project folder on the UNO Q:

```sh
mkdir -p ~/ArduinoApps/face-detection-robot
```

From your host computer, open up a new terminal, navigate into this directory, and run the following (replace <UNO_Q_IP_ADDRESS> with the IP address of your UNO Q board):

```sh
scp -r firmware/face-detection-robot/ arduino@<UNO_Q_IP_ADDRESS>:~/ArduinoApps
```

Now, we can use the Arduino App CLI to run and control apps. To start your program, run the following in the SSH terminal (connected to your UNO Q):

```sh
arduino-app-cli app start ~/ArduinoApps/face-detection-robot
```

You can check the logs with:

```sh
arduino-app-cli app logs ~/ArduinoApps/face-detection-robot
```

## License

All software, unless otherwise noted, is licensed under the [MIT License](https://opensource.org/license/mit).

>Copyright 2025 Shawn Hymel
>
>Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the “Software”), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:
>
>The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.
>
>THE SOFTWARE IS PROVIDED “AS IS”, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
