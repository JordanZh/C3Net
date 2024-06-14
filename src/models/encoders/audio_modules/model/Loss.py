import torch
import torch.nn as nn


class Loss_likeCLIP(nn.Module):
    def __init__(self, alpha=1.0, beta=1.0):
        super(Loss_likeCLIP, self).__init__()
        self.temperature = torch.tensor(0)
        self.alpha = alpha
        self.beta = beta

    def contrastive_loss(self, logits):  # contrastive loss function, adapted from
        return nn.functional.cross_entropy(logits, torch.arange(len(logits), device=logits.device))

    def forward(self, embedding1, embedding2):
        # normalized features
        embedding1 = embedding1 / embedding1.norm(p=2, dim=-1, keepdim=True)
        embedding2 = embedding2 / embedding2.norm(p=2, dim=-1, keepdim=True)

        logits = torch.matmul(embedding1, embedding2.t()) * torch.exp(self.temperature)

        loss1 = self.contrastive_loss(logits)
        loss2 = self.contrastive_loss(logits.t())

        return (self.alpha * loss1 + self.beta * loss2) / (self.alpha + self.beta)  # TODO: I don't know. Maybe I need to /batch_size ?


class Loss_N_Order(nn.Module):
    def __init__(self, order):
        super(Loss_N_Order, self).__init__()
        if order == 1:
            self.function = nn.L1Loss()
        elif order == 2:
            self.function = nn.MSELoss()

    def forward(self, embedding1, embedding2):
        #embedding1 = embedding1 / embedding1.norm(p=2, dim=-1, keepdim=True)
        #embedding2 = embedding2 / embedding2.norm(p=2, dim=-1, keepdim=True)

        return self.function(embedding1, embedding2)

