import argparse, os, sys, glob

sys.path.append(os.path.join(sys.path[0], '..'))

import torch
import numpy as np
from omegaconf import OmegaConf
from PIL import Image
from tqdm import tqdm, trange
from einops import rearrange
from torchvision.utils import make_grid

from ldm.util import instantiate_from_config
from ldm.models.diffusion.ddim import DDIMSampler
from ldm.models.diffusion.plms import PLMSSampler
from ldm.data.personalized import PersonalizedBase
from evaluation.clip_eval import LDMCLIPEvaluator

def load_model_from_config(config, ckpt, verbose=False):
    print(f"Loading model from {ckpt}")
    pl_sd = torch.load(ckpt, map_location="cpu")
    sd = pl_sd["state_dict"]
    model = instantiate_from_config(config.model)
    m, u = model.load_state_dict(sd, strict=False)
    if len(m) > 0 and verbose:
        print("missing keys:")
        print(m)
    if len(u) > 0 and verbose:
        print("unexpected keys:")
        print(u)

    model.cuda()
    model.eval()
    return model


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--prompt",
        type=str,
        nargs="?",
        default="a painting of a virus monster playing guitar",
        help="the prompt to render"
    )

    parser.add_argument(
        "--ckpt_path", 
        type=str, 
        default="/data/pretrained_models/ldm/text2img-large/model.ckpt", 
        help="Path to pretrained ldm text2img model")

    parser.add_argument(
        "--embedding_path", 
        type=str, 
        help="Path to a pre-trained embedding manager checkpoint")

    parser.add_argument(
        "--data_dir",
        type=str,
        help="Path to directory with images used to train the embedding vectors"
    )

    opt = parser.parse_args()


    config = OmegaConf.load("configs/latent-diffusion/txt2img-1p4B-eval_with_tokens.yaml")  # TODO: Optionally download from same location as ckpt and chnage this logic
    model = load_model_from_config(config, opt.ckpt_path)  # TODO: check path
    model.embedding_manager.load(opt.embedding_path)

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    model = model.to(device)

    evaluator = LDMCLIPEvaluator(device)

    prompt = opt.prompt

    data_loader = PersonalizedBase(opt.data_dir, size=256, flip_p=0.0)

    images = [torch.from_numpy(data_loader[i]["image"]).permute(2, 0, 1) for i in range(data_loader.num_images)]
    images = torch.stack(images, axis=0)

    sim_img, sim_text = evaluator.evaluate(model, images, opt.prompt)

    output_dir = os.path.join(opt.out_dir, prompt.replace(" ", "-"))

    print("Image similarity: ", sim_img)
    print("Text similarity: ", sim_text)