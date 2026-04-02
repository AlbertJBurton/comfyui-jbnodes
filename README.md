# ComfyUI JB Nodes

**NOTE: This project is still a beta project and is in active development. Expect changes to node definitions or functions that may be breaking changes.** 

A comprehensive suite of ComfyUI nodes for simulating analog black and white film photography and darkroom printing processes. These nodes provide physically accurate emulations of film stocks, spectral responses, Wratten filters, darkroom enlargers, film grain, and camera characteristics.

## Features

* **Accurate Film Emulation:** Simulates the spectral sensitivity and characteristic curves (H&D curves) of various classic black and white film stocks.
* **Spectral Processing:** Calculates the interaction between light sources, camera lenses, optical filters, and film emulsion using spectral data.
* **Darkroom Printing:** Emulates the process of printing a film negative onto photographic paper using a darkroom enlarger, supporting both graded and multigrade papers.
* **Optical Filters:** Includes a wide range of standard Wratten filters for modifying contrast and tonal response.
* **Film Grain:** Adds realistic, customizable film grain using GLSL shaders.
* **Camera & Aspect Ratios:** Emulates camera transmission characteristics and provides tools for cropping to standard film formats (e.g., 35mm, 6x6, 4x5).

## Installation

1. Navigate to your ComfyUI `custom_nodes` directory.
2. Clone this repository:
   ```bash
   git clone https://github.com/yourusername/comfyui-jbnodes.git
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

## Nodes Overview

### JB Film Development (`DeveloperLab`)
The core node for developing the virtual film negative. It takes an input image and processes it through the selected film stock, light source, developer, and optional camera/filters to produce a film negative. Supports advanced controls like push/pull processing and N-development.

### JB Film Stock (`FilmLab`)
Provides a selection of classic black and white film stocks (e.g., Kodak Tri-X, Ilford HP5) and their associated characteristic curves. Outputs the film stock data to be used by the Film Development node.

### JB Darkroom Enlarger (`PrintLabGraded` / `PrintLabMultigrade`)
Simulates printing the developed negative onto photographic paper. 
* **Graded:** Uses standard graded photographic papers.
* **Multigrade:** Simulates variable contrast papers, allowing you to use contrast filters to control the tonal range of the final print.

### JB Wratten Filter (`FilterLab`)
Applies standard optical filters (e.g., Yellow 8, Red 25) to the image before it hits the film, altering the spectral response and contrast of the resulting negative.

### JB Camera (`CameraLab`)
Simulates the spectral transmission characteristics of various camera lenses and systems.

### JB Film Grain (`FilmGrainLab`)
Applies realistic film grain to the image using a high-performance GLSL shader via `moderngl`.

### JB Crop Film Aspect Ratio & JB Film Aspect Ratio
Utility nodes for handling standard analog film formats (35mm, Medium Format 6x4.5, 6x6, 6x7, Large Format 4x5, 8x10, etc.) and cropping images to match these aspect ratios.

### JB Grayscale Image (`GrayscaleLab`)
A utility node for converting images to standard grayscale using accurate luminance calculations.

## Requirements

* `numpy`
* `torch`
* `colour-science`
* `moderngl`
* `Pillow`

See `requirements.txt` for details. Note: `moderngl` is required for the Film Grain node to function.

## License

This project is licensed under the terms specified in the `LICENSE` file.