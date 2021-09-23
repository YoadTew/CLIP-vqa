import torch
import clip.clip as clip
from vqa.image_dataset import VQAImageDataset
from tqdm import tqdm
from torch.utils.data import DataLoader
import sys
import pickle

VIS_BACKBONE = 'RN50'
DATA_FOLD = 'val'

device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load(VIS_BACKBONE, device=device)
image_id_to_features = {}

if sys.gettrace() is not None:
    N_WORKERS = 0
else:
    N_WORKERS = 4

dataset = VQAImageDataset('/home/work/Datasets/vqa2', preprocess, DATA_FOLD)
loader = DataLoader(dataset, 256, shuffle=False, num_workers=N_WORKERS)

for i, (image, image_id) in enumerate(tqdm(loader)):
    image = image.to(device)
    with torch.no_grad():
        image_features = model.encode_image(image).cpu().numpy()

    for j in range(image.size(0)):
        image_id_to_features[image_id[j].item()] = image_features[j]

with open(f'/home/work/Datasets/vqa2/CLIP_features/{VIS_BACKBONE}_{DATA_FOLD}.pkl', 'wb') as f:
    pickle.dump(image_id_to_features, f)



