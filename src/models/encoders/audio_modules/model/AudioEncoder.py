from model.ssast_model.ast_models import ASTModel
import torch.nn as nn
import torch.nn.functional as F

class AudioEncoder(nn.Module):
    def __init__(self, config):
        super(AudioEncoder, self).__init__()

        self.config = config
        ssast_config = config['ssast_config']
        self.unimodal_model = ASTModel(label_dim=ssast_config['label_dim'], fshape=ssast_config['fshape'], tshape=ssast_config['tshape'], fstride=ssast_config['fstride'], tstride=ssast_config['tstride'],
                                         input_fdim=ssast_config['input_fdim'], input_tdim=ssast_config['input_tdim'], model_size=ssast_config['model_size'],
                                         pretrain_stage=ssast_config['pretrain_stage'], load_pretrained_mdl_path=ssast_config['load_pretrained_mdl_path'])
        self.head = nn.Sequential(
            nn.PReLU(),
            nn.Linear(ssast_config['label_dim'], ssast_config['label_dim']),
            nn.PReLU(),
            nn.Linear(ssast_config['label_dim'], config['common_embedding_dim']),  # FIXME
        )


    def forward(self, x):
        x = self.unimodal_model.forward(x=x, task='ft_avgtok')

        x0 = x
        x = self.head(x)
        return x + x0

