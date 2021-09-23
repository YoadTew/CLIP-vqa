from torch.utils.data import Dataset
from collections import Counter, defaultdict
from torchvision import transforms
from PIL import Image
import torch
import json
import pickle
import glob

DATA_NAMES = {
    'train': {},
    'val': {
        'questions': 'questions/v2_OpenEnded_mscoco_val2014_questions.json',
        'annotations': 'annotations/v2_mscoco_val2014_annotations.json',
        'images': 'images/val2014'
    }
}

def extract_gt_answers(answers):
    gt_answers = [x['answer'] for x in answers]
    answers_count = Counter(gt_answers)
    answers_acc = [(k, min(answers_count[k] / 3, 1)) for k in answers_count]

    gt_answers = defaultdict(int)
    for k, v in answers_acc:
        gt_answers[k] = v

    return gt_answers

def question_caption_lambda(question, question_type):
    # if question_type == 'what are the':
    #     question = question[len('what are '):-1]
    #     return lambda x: f'{question} {x}'
    return lambda x: f'Question: {question} Answer: {x}'

class VQADataset(Dataset):
    def __init__(self, root_path, preprocess, clip_tokenize, data_type='train'):
        super(VQADataset, self).__init__()

        with open(f'{root_path}/{DATA_NAMES[data_type]["questions"]}') as f:
            data = json.load(f)
            self.questions = data['questions']

        with open(f'{root_path}/{DATA_NAMES[data_type]["annotations"]}') as f:
            data = json.load(f)
            self.annotations = data['annotations']

        with open(f'{root_path}/top_answers/top_answers_val.pkl', 'rb') as f:
            top_k_answers = pickle.load(f)
            self.top_k_answers = dict([(x[0], (x[1], x[2])) for x in top_k_answers])

        imgs_pathes = glob.glob(f'{root_path}/{DATA_NAMES[data_type]["images"]}/*.jpg')
        self.img_id_to_path = dict([(int(x.split('_')[-1].split('.')[0]), x) for x in imgs_pathes])

        self.clip_img_preprocess = preprocess
        self.clip_tokenize = clip_tokenize

    def __getitem__(self, index):

        question = self.questions[index]
        question_id = question['question_id']
        ques_text = question['question']

        img_path = self.img_id_to_path[question['image_id']]
        image = self.clip_img_preprocess(Image.open(img_path))

        annotation = self.annotations[index]
        ques_type = annotation['question_type']
        caption_transform = question_caption_lambda(ques_text, ques_type)
        possible_answers = self.top_k_answers[question_id][1]
        captions = [caption_transform(ans) for ans in possible_answers]
        captions = self.clip_tokenize(captions)

        gt_answers = extract_gt_answers(annotation['answers'])
        label = torch.tensor([gt_answers[x] for x in possible_answers], dtype=float)

        return captions, image, label

    def __len__(self):
        return len(self.questions)