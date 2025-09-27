from .ast_modules.ast_models import ASTModel
import torch
import torch.nn as nn
import torch.nn.functional as F

class AstEncoder(nn.Module):
    def __init__(self, pretrained_mdl_path, config=None):
        super(AstEncoder, self).__init__()

        self.config = config
        self.unimodal_model = ASTModel(label_dim=768,
                                       fshape=16, tshape=16, fstride=10, tstride=10,
                                       input_fdim=128, input_tdim=1024, model_size='base', 
                                       pretrain_stage=False, load_pretrained_mdl_path=pretrained_mdl_path)
        self.temperature = nn.Parameter(torch.tensor(0.07))

    def forward(self, x):
        x = self.unimodal_model.forward(x=x, task='ft_avgtok')
        x = F.normalize(x, dim=-1)
        return x

