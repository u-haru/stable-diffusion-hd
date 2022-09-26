import argparse, os, re
from email.mime import base
import torch
import numpy as np
from random import randint
from omegaconf import OmegaConf
from PIL import Image
from tqdm import tqdm, trange
from itertools import islice
from einops import rearrange
from torchvision.utils import make_grid
import time
from pytorch_lightning import seed_everything
from torch import autocast
from contextlib import contextmanager, nullcontext
from ldm.util import instantiate_from_config
from optimUtils import split_weighted_subprompts, logger
from transformers import logging
# from samplers import CompVisDenoiser
logging.set_verbosity_error()

# hd import
import subprocess
import gc
import PIL
from PIL import ImageDraw
from einops import repeat
import sys
# hd import end

# img2img func
def numpy_to_pil(images):
    """
    Convert a numpy image or a batch of images to a PIL image.
    """
    if images.ndim == 3:
        images = images[None, ...]
    images = (images * 255).round().astype("uint8")
    pil_images = [Image.fromarray(image) for image in images]

    return pil_images
# img2img func end


# hd func
def convert_pil_img(image):
    w, h = image.size
    w, h = map(lambda x: x - x % 32, (w, h))  # resize to integer multiple of 32
    image = image.resize((w, h), resample=PIL.Image.Resampling.LANCZOS)
    image = np.array(image).astype(np.float32) / 255.0
    image = image[None].transpose(0, 3, 1, 2)
    image = torch.from_numpy(image)
    return 2.*image - 1.

def addalpha(im, mask):
    imr, img, imb, ima = im.split()
    mmr, mmg, mmb, mma = mask.split()
    im = Image.merge('RGBA', [imr, img, imb, mma])  # we want the RGB from the original, but the transparency from the mask
    return(im)

# Alternative method composites a grid of images at the positions provided
def grid_merge(source, slices):
    source.convert("RGBA")
    for slice, posx, posy in slices: # go in reverse to get proper stacking
        source.alpha_composite(slice, (posx, posy))
    return source

def grid_coords(target, original, overlap):
    #generate a list of coordinate tuples for our sections, in order of how they'll be rendered
    #target should be the size for the gobig result, original is the size of each chunk being rendered
    center = []
    target_x, target_y = target
    center_x = int(target_x / 2)
    center_y = int(target_y / 2)
    original_x, original_y = original
    x = center_x - int(original_x / 2)
    y = center_y - int(original_y / 2)
    center.append((x,y)) #center chunk
    uy = y #up
    uy_list = []
    dy = y #down
    dy_list = []
    lx = x #left
    lx_list = []
    rx = x #right
    rx_list = []
    while uy > 0: #center row vertical up
        uy = uy - original_y + overlap
        uy_list.append((lx, uy))
    while (dy + original_y) <= target_y: #center row vertical down
        dy = dy + original_y - overlap
        dy_list.append((rx, dy))
    while lx > 0:
        lx = lx - original_x + overlap
        lx_list.append((lx, y))
        uy = y
        while uy > 0:
            uy = uy - original_y + overlap
            uy_list.append((lx, uy))
        dy = y
        while (dy + original_y) <= target_y:
            dy = dy + original_y - overlap
            dy_list.append((lx, dy))
    while (rx + original_x) <= target_x:
        rx = rx + original_x - overlap
        rx_list.append((rx, y))
        uy = y
        while uy > 0:
            uy = uy - original_y + overlap
            uy_list.append((rx, uy))
        dy = y
        while (dy + original_y) <= target_y:
            dy = dy + original_y - overlap
            dy_list.append((rx, dy))
    # calculate a new size that will fill the canvas, which will be optionally used in grid_slice and go_big
    last_coordx, last_coordy = dy_list[-1:][0]
    render_edgey = last_coordy + original_y # outer bottom edge of the render canvas
    render_edgex = last_coordx + original_x # outer side edge of the render canvas
    scalarx = render_edgex / target_x
    scalary = render_edgey / target_y
    if scalarx <= scalary:
        new_edgex = int(target_x * scalarx)
        new_edgey = int(target_y * scalarx)
    else:
        new_edgex = int(target_x * scalary)
        new_edgey = int(target_y * scalary)
    # now put all the chunks into one master list of coordinates (essentially reverse of how we calculated them so that the central slices will be on top)
    result = []
    for coords in dy_list[::-1]:
        result.append(coords)
    for coords in uy_list[::-1]:
        result.append(coords)
    for coords in rx_list[::-1]:
        result.append(coords)
    for coords in lx_list[::-1]:
        result.append(coords)
    result.append(center[0])
    return result, (new_edgex, new_edgey)

def get_resampling_mode():
    try:
        from PIL import __version__, Image
        major_ver = int(__version__.split('.')[0])
        if major_ver >= 9:
            return Image.Resampling.LANCZOS
        else:
            return Image.LANCZOS
    except Exception as ex:
        return 1  # 'Lanczos' irrespective of version.

# Chop our source into a grid of images that each equal the size of the original render
def grid_slice(source, overlap, og_size, maximize=False): 
    width, height = og_size # size of the slices to be rendered
    coordinates, new_size = grid_coords(source.size, og_size, overlap)
    if maximize == True:
        source = source.resize(new_size, get_resampling_mode()) # minor concern that we're resizing twice
        coordinates, new_size = grid_coords(source.size, og_size, overlap) # re-do the coordinates with the new canvas size
    # loc_width and loc_height are the center point of the goal size, and we'll start there and work our way out
    slices = []
    for coordinate in coordinates:
        x, y = coordinate
        slices.append(((source.crop((x, y, x+width, y+height))), x, y))
    global slices_todo
    slices_todo = len(slices) - 1
    return slices, new_size

def realesrgan2x(executable: str, input: str, output: str):
    process = subprocess.Popen([
        executable,
        '-i',
        input,
        '-o',
        output,
        '-n',
        'realesrgan-x4plus'
    ])
    process.wait()

    final_output = Image.open(output)
    final_output = final_output.resize((int(final_output.size[0] / 2), int(final_output.size[1] / 2)), get_resampling_mode())
    final_output.save(output)

def real(opt,base_filename):
    for _ in trange(opt.passes, desc="Passes"):
        realesrgan2x(opt.realesrgan, f"{base_filename}.png", f"{base_filename}u.png")
        base_filename = f"{base_filename}u"

        source_image = Image.open(f"{base_filename}.png")
        og_size = (opt.H,opt.W)
        slices, _ = grid_slice(source_image, opt.gobig_overlap, og_size, False)

        betterslices = []
        for _, chunk_w_coords in tqdm(enumerate(slices), "Slices"):
            chunk, coord_x, coord_y = chunk_w_coords
            modelFS.to(opt.device)
            init_image = convert_pil_img(chunk).to(opt.device)
            init_image = repeat(init_image, "1 ... -> b ...", b=batch_size)
            init_latent = modelFS.get_first_stage_encoding(modelFS.encode_first_stage(init_image))  # move to latent space

            assert 0. <= opt.strength <= 1., 'can only work with strength in [0.0, 1.0]'
            t_enc = int(opt.strength * opt.ddim_steps)
            print(f"target t_enc is {t_enc} steps")

            with torch.inference_mode():
                with precision_scope("cuda"):
                    for prompts in tqdm(data, desc="data"):
                        modelCS.to(opt.device)
                        uc = None
                        if opt.detail_scale != 1.0:
                            uc = modelCS.get_learned_conditioning(batch_size * [""])
                        if isinstance(prompts, tuple):
                            prompts = list(prompts)

                        subprompts, weights = split_weighted_subprompts(prompts[0])
                        if len(subprompts) > 1:
                            c = torch.zeros_like(uc)
                            totalWeight = sum(weights)
                            # normalize each "sub prompt" and add it
                            for i in range(len(subprompts)):
                                weight = weights[i]
                                # if not skip_normalize:
                                weight = weight / totalWeight
                                c = torch.add(c, modelCS.get_learned_conditioning(subprompts[i]), alpha=weight)
                        else:
                            c = modelCS.get_learned_conditioning(prompts)

                        if opt.device != "cpu":
                            mem = torch.cuda.memory_allocated(device=opt.device) / 1e6
                            modelCS.to("cpu")
                            while torch.cuda.memory_allocated(device=opt.device) / 1e6 >= mem:
                                time.sleep(1)

                        # encode (scaled latent)
                        z_enc = model.stochastic_encode(
                            init_latent,
                            torch.tensor([t_enc] * batch_size).to(opt.device),
                            opt.seed,
                            opt.ddim_eta,
                            opt.ddim_steps,
                        )
                        # decode it
                        samples_ddim = model.sample(
                            t_enc,
                            c,
                            z_enc,
                            unconditional_guidance_scale=opt.detail_scale,
                            unconditional_conditioning=uc,
                            sampler = "ddim",
                        )

                        modelFS.to(opt.device)

                        x_samples = modelFS.decode_first_stage(samples_ddim)
                        x_samples = torch.clamp((x_samples + 1.0) / 2.0, min=0.0, max=1.0)

                        for x_sample in x_samples:
                            x_sample = 255. * rearrange(x_sample.cpu().numpy(), 'c h w -> h w c')
                            resultslice = Image.fromarray(x_sample.astype(np.uint8)).convert('RGBA')
                            betterslices.append((resultslice.copy(), coord_x, coord_y))

                        if opt.device != "cpu":
                            mem = torch.cuda.memory_allocated() / 1e6
                            modelFS.to("cpu")
                            while torch.cuda.memory_allocated() / 1e6 >= mem:
                                time.sleep(1)
                        del samples_ddim
                        print("memory_final = ", torch.cuda.memory_allocated() / 1e6)


        alpha = Image.new('L', og_size, color=0xFF)
        alpha_gradient = ImageDraw.Draw(alpha)
        a = 0
        i = 0
        overlap = opt.gobig_overlap
        shape = (og_size, (0,0))
        while i < overlap:
            alpha_gradient.rectangle(shape, fill = a)
            a += 4
            i += 1
            shape = ((og_size[0] - i, og_size[1]- i), (i,i))
        mask = Image.new('RGBA', og_size, color=0)
        mask.putalpha(alpha)
        finished_slices = []
        for betterslice, x, y in betterslices:
            finished_slice = addalpha(betterslice, mask)
            finished_slices.append((finished_slice, x, y))
        # # Once we have all our images, use grid_merge back onto the source, then save
        final_output = grid_merge(source_image.convert("RGBA"), finished_slices).convert("RGB")
        final_output.save(f"{base_filename}d.png")
        base_filename = f"{base_filename}d"

        torch.cuda.empty_cache()
        gc.collect()
    
    # put_watermark(final_output, wm_encoder)
    final_output.save(f"{base_filename}.png")
# hd func end


def chunk(it, size):
    it = iter(it)
    return iter(lambda: tuple(islice(it, size)), ())


def load_model_from_config(ckpt, verbose=False):
    print(f"Loading model from {ckpt}")
    pl_sd = torch.load(ckpt, map_location="cpu")
    if "global_step" in pl_sd:
        print(f"Global Step: {pl_sd['global_step']}")
    sd = pl_sd["state_dict"]
    return sd


config = "optimizedSD/v1-inference.yaml"
DEFAULT_CKPT = "models/ldm/stable-diffusion-v1/model.ckpt"

parser = argparse.ArgumentParser()

parser.add_argument(
    "--prompt", type=str, nargs="?", default="a painting of a virus monster playing guitar", help="the prompt to render"
)
parser.add_argument("--outdir", type=str, nargs="?", help="dir to write results to", default="outputs/txt2imghd-samples")
parser.add_argument(
    "--skip_grid",
    action="store_true",
    help="do not save a grid, only individual samples. Helpful when evaluating lots of samples",
)
parser.add_argument(
    "--skip_save",
    action="store_true",
    help="do not save individual samples. For speed measurements.",
)
parser.add_argument(
    "--ddim_steps",
    type=int,
    default=50,
    help="number of ddim sampling steps",
)

parser.add_argument(
    "--fixed_code",
    action="store_true",
    help="if enabled, uses the same starting code across samples ",
)
parser.add_argument(
    "--ddim_eta",
    type=float,
    default=0.0,
    help="ddim eta (eta=0.0 corresponds to deterministic sampling",
)
parser.add_argument(
    "--n_iter",
    type=int,
    default=1,
    help="sample this often",
)
parser.add_argument(
    "--H",
    type=int,
    default=512,
    help="image height, in pixel space",
)
parser.add_argument(
    "--W",
    type=int,
    default=512,
    help="image width, in pixel space",
)
parser.add_argument(
    "--C",
    type=int,
    default=4,
    help="latent channels",
)
parser.add_argument(
    "--f",
    type=int,
    default=8,
    help="downsampling factor",
)
# parser.add_argument(
#     "--n_samples",
#     type=int,
#     default=1,
#     help="how many samples to produce for each given prompt. A.k.a. batch size",
# )
parser.add_argument(
    "--n_rows",
    type=int,
    default=0,
    help="rows in the grid (default: n_samples)",
)
parser.add_argument(
    "--scale",
    type=float,
    default=7.5,
    help="unconditional guidance scale: eps = eps(x, empty) + scale * (eps(x, cond) - eps(x, empty))",
)
parser.add_argument(
    "--device",
    type=str,
    default="cuda",
    help="specify GPU (cuda/cuda:0/cuda:1/...)",
)
parser.add_argument(
    "--from-file",
    type=str,
    help="if specified, load prompts from this file",
)
parser.add_argument(
    "--seed",
    type=int,
    default=None,
    help="the seed (for reproducible sampling)",
)
parser.add_argument(
    "--unet_bs",
    type=int,
    default=1,
    help="Slightly reduces inference time at the expense of high VRAM (value > 1 not recommended )",
)
parser.add_argument(
    "--turbo",
    action="store_true",
    help="Reduces inference time on the expense of 1GB VRAM",
)
parser.add_argument(
    "--precision", 
    type=str,
    help="evaluate at this precision",
    choices=["full", "autocast"],
    default="autocast"
)
parser.add_argument(
    "--format",
    type=str,
    help="output image format",
    choices=["jpg", "png"],
    default="png",
)
parser.add_argument(
    "--sampler",
    type=str,
    help="sampler",
    choices=["ddim", "plms","heun", "euler", "euler_a", "dpm2", "dpm2_a", "lms"],
    default="plms",
)
parser.add_argument(
    "--ckpt",
    type=str,
    help="path to checkpoint of model",
    default=DEFAULT_CKPT,
)

# hd params
parser.add_argument(
    "--strength",
    type=float,
    default=0.3,
    help="strength for noising/unnoising. 1.0 corresponds to full destruction of information in init image",
)
parser.add_argument(
    "--passes",
    type=int,
    default=1,
    help="number of upscales/details",
)
parser.add_argument(
    "--realesrgan",
    type=str,
    default="realesrgan-ncnn-vulkan/realesrgan-ncnn-vulkan",
    help="path to realesrgan executable"
)
parser.add_argument(
    "--detail_scale",
    type=float,
    default=10,
    help="unconditional guidance scale when detailing: eps = eps(x, empty) + scale * (eps(x, cond) - eps(x, empty))",
)
parser.add_argument(
    "--img",
    type=str,
    nargs="?",
    help="only do detailing, using this path (will be copied to output dir)"
)
parser.add_argument(
    "--gobig_overlap",
    type=int,
    default=128,
    help="overlap size for GOBIG",
)
# hd params end

opt = parser.parse_args()
opt.n_samples=1

tic = time.time()
os.makedirs(opt.outdir, exist_ok=True)
outpath = opt.outdir
grid_count = len(os.listdir(outpath)) - 1

if opt.seed == None:
    opt.seed = randint(0, 1000000)
seed_everything(opt.seed)

# Logging
logger(vars(opt), log_csv = "logs/txt2img_logs.csv")

sd = load_model_from_config(f"{opt.ckpt}")
li, lo = [], []
for key, value in sd.items():
    sp = key.split(".")
    if (sp[0]) == "model":
        if "input_blocks" in sp:
            li.append(key)
        elif "middle_block" in sp:
            li.append(key)
        elif "time_embed" in sp:
            li.append(key)
        else:
            lo.append(key)
for key in li:
    sd["model1." + key[6:]] = sd.pop(key)
for key in lo:
    sd["model2." + key[6:]] = sd.pop(key)

config = OmegaConf.load(f"{config}")

model = instantiate_from_config(config.modelUNet)
_, _ = model.load_state_dict(sd, strict=False)
model.eval()
model.unet_bs = opt.unet_bs
model.cdevice = opt.device
model.turbo = opt.turbo

modelCS = instantiate_from_config(config.modelCondStage)
_, _ = modelCS.load_state_dict(sd, strict=False)
modelCS.eval()
modelCS.cond_stage_model.device = opt.device

modelFS = instantiate_from_config(config.modelFirstStage)
_, _ = modelFS.load_state_dict(sd, strict=False)
modelFS.eval()
del sd

if opt.device != "cpu" and opt.precision == "autocast":
    model.half()
    modelCS.half()

start_code = None
if opt.fixed_code:
    start_code = torch.randn([opt.n_samples, opt.C, opt.H // opt.f, opt.W // opt.f], device=opt.device)


batch_size = opt.n_samples
n_rows = opt.n_rows if opt.n_rows > 0 else batch_size
if not opt.from_file:
    assert opt.prompt is not None
    prompt = opt.prompt
    print(f"Using prompt: {prompt}")
    data = [batch_size * [prompt]]

else:
    print(f"reading prompts from {opt.from_file}")
    with open(opt.from_file, "r") as f:
        text = f.read()
        print(f"Using prompt: {text.strip()}")
        data = text.splitlines()
        data = batch_size * list(data)
        data = list(chunk(sorted(data), batch_size))


if opt.precision == "autocast" and opt.device != "cpu":
    precision_scope = autocast
else:
    precision_scope = nullcontext

seeds = ""

# hd img
if opt.img is not None:
    sample_path = os.path.join(outpath, "_".join(re.split(":| ", opt.prompt)))[:150]
    os.makedirs(sample_path, exist_ok=True)
    base_count = len(os.listdir(sample_path))

    img = Image.open(opt.img).convert("RGB")
    base_filename = os.path.join(sample_path, f"{base_count:05}")
    img.save(base_filename+"."+opt.format)
    real(opt,base_filename)

    toc = time.time()
    time_taken = (toc - tic) / 60.0

    print(
        (
            "Samples finished in {0:.2f} minutes and exported to "
            + sample_path
            + "\n Seeds used = "
            + seeds[:-1]
        ).format(time_taken)
    )
    sys.exit(0)
# hd img end

with torch.no_grad():

    all_samples = list()
    for n in trange(opt.n_iter, desc="Sampling"):
        for prompts in tqdm(data, desc="data"):

            sample_path = os.path.join(outpath, "_".join(re.split(":| ", prompts[0])))[:150]
            os.makedirs(sample_path, exist_ok=True)
            base_count = len(os.listdir(sample_path))

            with precision_scope("cuda"):
                modelCS.to(opt.device)
                uc = None
                if opt.scale != 1.0:
                    uc = modelCS.get_learned_conditioning(batch_size * [""])
                if isinstance(prompts, tuple):
                    prompts = list(prompts)

                subprompts, weights = split_weighted_subprompts(prompts[0])
                if len(subprompts) > 1:
                    c = torch.zeros_like(uc)
                    totalWeight = sum(weights)
                    # normalize each "sub prompt" and add it
                    for i in range(len(subprompts)):
                        weight = weights[i]
                        # if not skip_normalize:
                        weight = weight / totalWeight
                        c = torch.add(c, modelCS.get_learned_conditioning(subprompts[i]), alpha=weight)
                else:
                    c = modelCS.get_learned_conditioning(prompts)

                shape = [opt.n_samples, opt.C, opt.H // opt.f, opt.W // opt.f]

                if opt.device != "cpu":
                    mem = torch.cuda.memory_allocated() / 1e6
                    modelCS.to("cpu")
                    while torch.cuda.memory_allocated() / 1e6 >= mem:
                        time.sleep(1)

                samples_ddim = model.sample(
                    S=opt.ddim_steps,
                    conditioning=c,
                    seed=opt.seed,
                    shape=shape,
                    verbose=False,
                    unconditional_guidance_scale=opt.scale,
                    unconditional_conditioning=uc,
                    eta=opt.ddim_eta,
                    x_T=start_code,
                    sampler = opt.sampler,
                )

                modelFS.to(opt.device)

                print(samples_ddim.shape)
                print("saving images")
                for i in range(batch_size):

                    x_samples_ddim = modelFS.decode_first_stage(samples_ddim[i].unsqueeze(0))
                    x_sample = torch.clamp((x_samples_ddim + 1.0) / 2.0, min=0.0, max=1.0)
                    x_sample = 255.0 * rearrange(x_sample[0].cpu().numpy(), "c h w -> h w c")
                    # Image.fromarray(x_sample.astype(np.uint8)).save(
                    #    os.path.join(sample_path, "seed_" + str(opt.seed) + "_" + f"{base_count:05}.{opt.format}")
                    # )
                    
                    # hd
                    base_filename = os.path.join(sample_path, "seed_" + str(opt.seed) + "_" + f"{base_count:05}")
                    Image.fromarray(x_sample.astype(np.uint8)).save(base_filename+"."+opt.format)
                    real(opt,base_filename)
                    # hd end
                    seeds += str(opt.seed) + ","
                    opt.seed += 1
                    base_count += 1

                if opt.device != "cpu":
                    mem = torch.cuda.memory_allocated() / 1e6
                    modelFS.to("cpu")
                    while torch.cuda.memory_allocated() / 1e6 >= mem:
                        time.sleep(1)
                del samples_ddim
                print("memory_final = ", torch.cuda.memory_allocated() / 1e6)

toc = time.time()

time_taken = (toc - tic) / 60.0

print(
    (
        "Samples finished in {0:.2f} minutes and exported to "
        + sample_path
        + "\n Seeds used = "
        + seeds[:-1]
    ).format(time_taken)
)
