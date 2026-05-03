# ComfyUI-JBNodes

A comprehensive suite of ComfyUI nodes for simulating analog black-and-white film photography and darkroom printing processes. See the [Wiki](https://github.com/AlbertJBurton/comfyui-jbnodes/wiki) for details.

> _**NOTE: This project is still a beta project and is in active development. Expect changes to node definitions or functions that may be breaking changes to previous versions.**_

## Dependencies

* [`numpy`](https://github.com/numpy/numpy)
* [`torch`](https://github.com/pytorch/pytorch)
* [`colour-science`](https://github.com/colour-science/colour)
* [`moderngl`](https://github.com/moderngl/moderngl)
* [`Pillow`](https://github.com/python-pillow/Pillow)

## Installation

1. Navigate to your ComfyUI `custom_nodes` directory.

2. Clone this repository:
   ```bash
   git clone https://github.com/AlbertJBurton/comfyui-jbnodes.git
   ```
3. Navigate into the cloned directory:
   ```bash
   cd comfyui-jbnodes
   ```
4. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```
5. Restart ComfyUI.

## Installation Issues and Fixes

### Issue Installing ModernGL Library

Some systems have demonstrated issues when trying to install the `moderngl` library used to process the film grain algorithm. Users running a fresh install of `Ubunutu 26.04 LTS` are seeing this issue. You may see the following error, or something similar, in the log during ComfyUI startup.

```bash
glcontext/x11.cpp:5:10: fatal error: X11/Xlib.h: No such file or directory
      5 | #include <X11/Xlib.h>
        |          ^~~~~~~~~~~~
      compilation terminated.
      error: command '/usr/bin/x86_64-linux-gnu-g++' failed with exit code 1
```
The error you are seeing occurs when `pip` tries to build the `glcontext` wheel from source, it requires the X11 development headers to compile the C++ code, but they aren't present on your system. To resolve this issue, you need to install the `libx11-dev` package. Since `moderngl` often requires additional OpenGL headers, it is a good idea to install the basic Mesa development utilities as well.

Run the following command in your terminal:

```bash
sudo apt update
sudo apt install libx11-dev libgl1-mesa-dev
```

Once you have installed the system packages, try the installation again within your virtual environment:

```bash
pip install moderngl
```

If you encountered a similar error regarding `GL/gl.h` in the build process, the `libgl1-mesa-dev` package included in the `apt install` command above should already have that covered.

### Installation for Runpod Users

When using ComfyUI-JBNodes film grain nodes in a RunPod environment, you may see the following in the ComfyUI log.

```bash
[comfyui-jbnodes] EGL context failed, falling back to auto-detect: libEGL.so not loaded
[comfyui-jbnodes] Failed to initialize moderngl context: (standalone) XOpenDisplay: cannot open display
```

This is happening because you are running an optimized "slim" container on RunPod. These slim Docker images strip out system-level rendering libraries to save space. Because moderngl hardware-accelerates the film grain shader on the GPU, it needs the base OpenGL/EGL libraries to create a headless rendering context, which are missing from your OS.

To get the film grain node working, open a terminal in your workspace/RunPod and install the required system libraries:

```bash
apt-get update
apt-get install -y libgl1 libegl1
```

Once installed, restart ComfyUI, and moderngl will be able to bind to the GPU to hardware-accelerate the film grain shader.

## License

This project is licensed under the terms specified in the `LICENSE` file.
