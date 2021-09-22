import json
import glob
import cv2
import matplotlib.pyplot as plt
from PIL import Image

import torch
from torch.utils.data import DataLoader
import torch.nn.functional as F
from torch import nn, optim

import clip.clip as clip
from clip import model as clip_model

EPOCHS = 5
BATCH_SIZE = 256

train_dataloader = DataLoader()

# https://github.com/openai/CLIP/issues/57
def convert_models_to_fp32(model):
    for p in model.parameters():
        p.data = p.data.float()
        p.grad.data = p.grad.data.float()


device = "cuda:0" if torch.cuda.is_available() else "cpu"  # If using GPU then use mixed precision training.
model, preprocess = clip.load("RN50", device=device, jit=False)  # Must set jit=False for training
if device == "cpu":
    model.float()
else:
    clip_model.convert_weights(model)  # Actually this line is unnecessary since clip by default already on float16

loss_img = nn.CrossEntropyLoss()
loss_txt = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.soft_prompts.parameters(), lr=5e-5,betas=(0.9,0.98),eps=1e-6,weight_decay=0.2) #Params used from paper, the lr is smaller, more safe for fine tuning to new dataset

for epoch in range(EPOCHS):
    for batch in train_dataloader:
        optimizer.zero_grad()

