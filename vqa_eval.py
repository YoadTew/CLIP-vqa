import json
import glob
import torch
# import clip.clip as clip
import clip
import pickle
from collections import Counter, defaultdict
from tqdm import tqdm
from torch.utils.data import DataLoader
import sys

from ban.vqa_dataset import VQADataset


print(clip.available_models())
device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("RN50", device=device, download_root='/home/work/checkpoints/CLIP')

with open('/home/work/Datasets/vqa2/questions/v2_OpenEnded_mscoco_val2014_questions.json') as f:
    data = json.load(f)
    questions = data['questions']

with open('/home/work/Datasets/vqa2/annotations/v2_mscoco_val2014_annotations.json') as f:
    data = json.load(f)
    annotations = data['annotations']

with open('/home/work/Datasets/vqa2/top_answers/top_answers_val.pkl', 'rb') as f:
    top_k_answers = pickle.load(f)
    top_k_answers = dict([(x[0], (x[1], x[2])) for x in top_k_answers])

imgs_pathes = glob.glob('/home/work/Datasets/vqa2/images/val2014/*.jpg')
img_id_to_path = dict([(int(x.split('_')[-1].split('.')[0]), x) for x in imgs_pathes])

QUESTION_TYPES = ['what are the']

def clip_infer(image, text):
    with torch.no_grad():
        image_features = model.encode_image(image)

        b, k, n = text.size()
        text = text.view(b*k, n)
        text_features = model.encode_text(text)
        text_features = text_features.view(b, k, -1)

        # normalized features
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)

        # cosine similarity as logits
        logit_scale = model.logit_scale.exp()
        logits_per_image = logit_scale * torch.bmm(image_features.unsqueeze(1), text_features.permute(0,2,1)).squeeze(1)

        probs = logits_per_image.softmax(dim=-1).cpu()

        return probs

def question_caption_lambda(question, question_type):
    # if question_type == 'what are the':
    #     question = question[len('what are '):-1]
    #     return lambda x: f'{question} {x}'
    # else:
    #     return lambda x: f'Question: {question} Answer: {x}'
    return lambda x: f'Question: {question} Answer: {x}'

def extract_gt_answers(answers):
    gt_answers = [x['answer'] for x in answers]
    answers_count = Counter(gt_answers)
    answers_acc = [(k, min(answers_count[k] / 3, 1)) for k in answers_count]

    gt_answers = defaultdict(int)
    for k, v in answers_acc:
        gt_answers[k] = v

    return gt_answers


def main():
    type_to_question = {}

    TP = 0
    upper_bound_accuracy = 0
    n_samples = 0

    if sys.gettrace() is not None:
        print('DEBUG!')
        N_WORKERS = 0
    else:
        print('NOT DEBUG!')
        N_WORKERS = 4

    dataset = VQADataset(questions, annotations, top_k_answers, img_id_to_path, preprocess, clip.tokenize)
    loader = DataLoader(dataset, 256, shuffle=False, num_workers=N_WORKERS)

    for i, (text, image, label) in enumerate(tqdm(loader)):
        image = image.to(device)
        text = text.to(device)

        upper_bound_accuracy += label.max(dim=1).values.sum().item()

        probs = clip_infer(image, text)
        pred_answer = torch.argmax(probs, dim=1)

        TP += label[torch.arange(256), pred_answer].sum().item()
        n_samples += image.size(0)

        if i == 30:
            break

    print(f'TP: {TP}, Accuracy: {TP/n_samples}, Upper bound: {upper_bound_accuracy / n_samples}')

main()