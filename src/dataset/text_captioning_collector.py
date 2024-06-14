import csv
import json
import numpy as np
import torch
from transformers import AutoTokenizer, CLIPTextModelWithProjection


class CaptionCollector:
    def __init__(self):
        self.task2path = {
            'coco17': 'COCO_17/coco_ann2017/annotations/captions_train2017.json',
            'audioCap': 'AudioCap/train.csv',
            'audioSet': 'AudioSet/audioset_unbalanced_train.csv',
        }
        self.caption = []
        self.embedding = None

    def collect_coco(self):
        with open(self.task2path['coco17']) as json_file:
            caption_dict = json.load(json_file)
            caption_dict_list = caption_dict['annotations']

        for one_dict in caption_dict_list:
            self.caption.append(one_dict['caption'])
        print(len(self.caption))

    def collect_audioCap(self):
        with open(self.task2path['audioCap']) as csvfile:
            spamreader = csv.reader(csvfile, delimiter=',')
            header = True
            for row in spamreader:
                if header:
                    header = False
                    continue
                self.caption.append(row[3])
        print(len(self.caption))

    def collect_audioSet(self):
        with open(self.task2path['audioSet']) as csvfile:
            spamreader = csv.reader(csvfile, delimiter=',')
            for row in spamreader:
                self.caption.append(row[2])
        print(len(self.caption))
        #print(self.caption)

    def collect_all(self):
        self.collect_coco()
        self.collect_audioCap()
        self.collect_audioSet()

        with open('text_captioning/caption.txt', 'w') as f:
            for line in self.caption:
                f.write(f"{line}\n")

    def compute_CLIP_embedding(self, clip_version='openai/clip-vit-large-patch14', device='cuda'):
        caption_embedding = np.zeros((len(self.caption), 768))

        model = CLIPTextModelWithProjection.from_pretrained(clip_version)
        tokenizer = AutoTokenizer.from_pretrained(clip_version)
        model = model.to(device)

        bs = 32
        for i in range(len(self.caption) // bs):
            text_list = self.caption[bs * i: bs * (i + 1)]

            inputs = tokenizer(text_list, padding=True, return_tensors='pt')
            inputs = inputs.to(device)
            outputs = model(**inputs)
            part_text_embeds = outputs.text_embeds
            caption_embedding[bs * i: bs * (i + 1)] = part_text_embeds.detach().cpu().numpy()
            print(i, (len(self.caption) // bs), f'  @{i/(len(self.caption) // bs)*100}%')

        self.embedding = caption_embedding
        np.save('text_captioning/embedding0.npy', self.embedding)
        print(caption_embedding[:50])


if __name__ == '__main__':
    collector = CaptionCollector()
    collector.collect_all()
    collector.compute_CLIP_embedding()



