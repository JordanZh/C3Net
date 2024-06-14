from transformers import AutoTokenizer, CLIPTextModelWithProjection
import torch.nn as nn

class TextEncoder(nn.Module):
    def __init__(self, device):
        super(TextEncoder, self).__init__()

        self.tokenizer = AutoTokenizer.from_pretrained("openai/clip-vit-large-patch14")
        self.model = CLIPTextModelWithProjection.from_pretrained("openai/clip-vit-large-patch14")
        self.device = device

    def forward(self, text_list):
        inputs = self.tokenizer(text_list, padding=True, return_tensors="pt")
        inputs = inputs.to(self.device)
        outputs = self.model(**inputs)
        # TODO: .to(device) ?

        # last_hidden_state = outputs.last_hidden_state
        return outputs.text_embeds

