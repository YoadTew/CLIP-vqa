from torch.utils.data import Dataset
from collections import Counter, defaultdict
from torchvision import transforms
from PIL import Image
import torch

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
    # else:
    #     return lambda x: f'Question: {question} Answer: {x}'
    return lambda x: f'Question: {question} Answer: {x}'

class VQADataset(Dataset):
    def __init__(self, questions, annotations, top_k_answers, img_id_to_path, preprocess, clip_tokenize):
        super(VQADataset, self).__init__()
        self.questions = questions
        self.annotations = annotations
        self.top_k_answers = top_k_answers
        self.img_id_to_path = img_id_to_path

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