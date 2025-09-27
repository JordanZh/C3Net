import torch.nn as nn
import torch
from transformers import AutoTokenizer, CLIPTextModelWithProjection


class TextEncoder_(nn.Module):
    def __init__(self, input_device):
        super(TextEncoder, self).__init__()

        self.tokenizer = AutoTokenizer.from_pretrained("openai/clip-vit-large-patch14")
        self.model = CLIPTextModelWithProjection.from_pretrained("openai/clip-vit-large-patch14")
        self.device = input_device
        self.max_length = 77

    def forward(self, text_list):
        inputs = self.tokenizer(text_list, padding=True, return_tensors="pt")
        inputs = inputs.to(self.device)
        outputs = self.model(**inputs)

        return outputs.text_embeds


from .clip_modules import CLIPModel, CLIPTokenizer, CLIPConfig
class AbstractEncoder__(nn.Module):
    def __init__(self):
        super().__init__()

    def encode(self, *args, **kwargs):
        raise NotImplementedError

class TextEncoder(AbstractEncoder__):
    def __init__(self,  input_device,
                 version="openai/clip-vit-large-patch14", 
                 max_length=77):
        super().__init__()
        self.tokenizer = CLIPTokenizer.from_pretrained(version)
        config = CLIPConfig.from_pretrained(version)
        self.model = CLIPModel(config, add_temporal_attention=True)
        self.max_length = max_length
        self.model.vision_model = None
        self.model.visual_projection = None
        self.freeze()

    def freeze(self):
        self.model = self.model.eval()
        for param in self.parameters():
            param.requires_grad = False

    def get_device(self):
        return next(self.parameters()).device

    def forward(self, text):
        batch_encoding = self.tokenizer(text, truncation=True, max_length=self.max_length, return_length=True,
                                        return_overflowing_tokens=False, padding="max_length", return_tensors="pt")
        tokens = batch_encoding["input_ids"].to(self.get_device())

        outputs = self.model.text_model(input_ids=tokens)
        z_pooled = outputs.pooler_output
        z_pooled = self.model.text_projection(z_pooled)
        z_pooled = z_pooled / torch.norm(z_pooled, dim=-1, keepdim=True)
        return z_pooled

