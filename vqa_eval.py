import json
import random
import glob
import torch
import numpy as np
import clip.clip as clip
import pickle
from collections import Counter, defaultdict
from tqdm import tqdm
from torch.utils.data import DataLoader
import sys

from vqa.vqa_dataset import VQADataset

SOFT_PROMPT = True
ITER_TO_BREAK = 999

def eval_init():
    global model, preprocess, device

    torch.manual_seed(42)
    np.random.seed(42)
    random.seed(42)
    print(clip.available_models())
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f'Using device: {device}')
    model, preprocess = clip.load("RN50", device=device, download_root='/home/work/checkpoints/CLIP')

def clip_infer(image, text):
    with torch.no_grad():
        image_features = model.encode_image(image)

        b, k, n = text.size()
        text = text.view(b*k, n)
        text_features = model.encode_text(text, soft_prompting=SOFT_PROMPT)
        text_features = text_features.view(b, k, -1)

        # normalized features
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)

        # cosine similarity as logits
        logit_scale = model.logit_scale.exp()
        logits_per_image = logit_scale * torch.bmm(image_features.unsqueeze(1), text_features.permute(0,2,1)).squeeze(1)

        probs = logits_per_image.softmax(dim=-1).cpu()

        return probs

def main():
    eval_init()

    TP = 0
    upper_bound_accuracy = 0
    n_samples = 0

    if sys.gettrace() is not None:
        N_WORKERS = 0
    else:
        N_WORKERS = 4

    dataset = VQADataset('/home/work/Datasets/vqa2', preprocess, clip.tokenize, 'val')
    loader = DataLoader(dataset, 256, shuffle=False, num_workers=N_WORKERS)

    for i, (text, image, label) in enumerate(tqdm(loader)):
        image = image.to(device)
        text = text.to(device)

        upper_bound_accuracy += label.max(dim=1).values.sum().item()

        probs = clip_infer(image, text)
        pred_answer = torch.argmax(probs, dim=1)

        TP += label[torch.arange(256), pred_answer].sum().item()
        n_samples += image.size(0)

        if i == ITER_TO_BREAK:
            break

    print(f'TP: {TP}, Accuracy: {TP/n_samples}, Upper bound: {upper_bound_accuracy / n_samples}')

main()