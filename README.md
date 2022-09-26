# Optimized txt2imghd

This repo is a modified version of the Optimized Stable Diffusion repo, optimized to generate high-resolution image with [Real-ESRGAN](https://github.com/xinntao/Real-ESRGAN) as the upscaler.  
Some of code were quoted from [jquesnelle/txt2imghd](https://github.com/jquesnelle/txt2imghd) (sorry and thank you).

Modified file is only [optimizedSD/optimized_txt2imghd.py](optimizedSD/optimized_txt2imghd.py).

## Installation

1. If you have already cloned the original repository, you can just download and copy [optimized_txt2imghd.py](optimizedSD/optimized_txt2imghd.py) instead of cloning the repo. If not, you can also clone this repo and follow the same installation steps as the [original](https://github.com/basujindal/stable-diffusion) (mainly creating the conda environment and placing the weights at the specified location).

2. Download [Real-ESRGAN](https://github.com/xinntao/Real-ESRGAN/releases) (the respective `realesrgan-ncnn-vulkan` .zip for your OS) and unzip it into the root of repository (must be able to use ./realesrgan-ncnn-vulkan/realesrgan-ncnn-vulkan).



## Usage

optimized_txt2imghd has most of the same parameters as optimized_txt2img. `n_samples` parameter has been removed. The `strength` parameter controls how much detailing to do (between 0.0-1.0). `passes` parameter controls the number of upscaling.

sample:

```bash
python optimizedSD/optimized_txt2imghd.py --prompt "something you want" \
 --n_iter 2 --passes 2
```



## Added params

`--passes` number of upscaling/detailing passes  
default = 1

`--strength` strength for noising/unnoising. 1.0 corresponds to full destruction of information in init image (especially useful when using an existing image)   
default = 0.3

`--realesrgan` path to realesrgan executable  
default = realesrgan-ncnn-vulkan/realesrgan-ncnn-vulkan

`--img` only do detailing, using this path (will be copied to output dir)  
default = None
