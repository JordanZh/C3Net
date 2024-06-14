import torch.nn as nn
from timm.models import create_model
import json
import torch
from collections import OrderedDict
from argparse import ArgumentParser

from model.VideoMAE_v1 import modeling_finetune

class VideoEncoder_v1(nn.Module):
    def __init__(self, config):
        super(VideoEncoder_v1, self).__init__()
        self.config = config
        self.video_mae_configs = config['video_mae_configs']

        self.helper_args = ArgumentParser().parse_args()
        with open(self.video_mae_configs['helper_arg_path'], 'r') as f:
            self.helper_args.__dict__ = json.load(f)
        args = self.helper_args

        self.unimodal_model = create_model(
            args.model,
            pretrained=False,
            num_classes=args.nb_classes,
            all_frames=args.num_frames * args.num_segments,
            tubelet_size=args.tubelet_size,
            fc_drop_rate=args.fc_drop_rate,
            drop_rate=args.drop,
            drop_path_rate=args.drop_path,
            attn_drop_rate=args.attn_drop_rate,
            drop_block_rate=None,
            use_checkpoint=args.use_checkpoint,
            use_mean_pooling=args.use_mean_pooling,
            init_scale=args.init_scale,
        )
        self.load_and_init_unimodal_model()

        self.head = nn.Sequential(
            nn.Linear(768, config['common_embedding_dim']),
        )

    def forward(self, x):
        x = self.unimodal_model(x)
        x0 = x
        x = self.head(x)
        return x + x0

    def load_and_init_unimodal_model(self):
        args = self.helper_args

        args.window_size = 'fix arg.windows_size'
        args.patch_size = 'fix arg.batch_size'

        if args.finetune:
            if args.finetune.startswith('https'):
                checkpoint = torch.hub.load_state_dict_from_url(
                    args.finetune, map_location='cpu', check_hash=True)
            else:
                checkpoint = torch.load(args.finetune, map_location='cpu')

            print("Load ckpt from %s" % args.finetune)
            checkpoint_model = None
            for model_key in args.model_key.split('|'):
                if model_key in checkpoint:
                    checkpoint_model = checkpoint[model_key]
                    print("Load state_dict by model_key = %s" % model_key)
                    break
            if checkpoint_model is None:
                checkpoint_model = checkpoint
            state_dict = self.unimodal_model.state_dict()
            for k in ['head.weight', 'head.bias']:
                if k in checkpoint_model and checkpoint_model[k].shape != state_dict[k].shape:
                    print(f"Removing key {k} from pretrained checkpoint")
                    del checkpoint_model[k]

            all_keys = list(checkpoint_model.keys())
            new_dict = OrderedDict()
            for key in all_keys:
                if key.startswith('backbone.'):
                    new_dict[key[9:]] = checkpoint_model[key]
                elif key.startswith('encoder.'):
                    new_dict[key[8:]] = checkpoint_model[key]
                else:
                    new_dict[key] = checkpoint_model[key]
            checkpoint_model = new_dict

            # interpolate position embedding
            if 'pos_embed' in checkpoint_model:
                pos_embed_checkpoint = checkpoint_model['pos_embed']
                embedding_size = pos_embed_checkpoint.shape[-1]  # channel dim
                num_patches = self.unimodal_model.patch_embed.num_patches  #
                num_extra_tokens = self.unimodal_model.pos_embed.shape[-2] - num_patches  # 0/1

                # height (== width) for the checkpoint position embedding
                orig_size = int(((pos_embed_checkpoint.shape[-2] - num_extra_tokens) // (
                            args.num_frames // self.unimodal_model.patch_embed.tubelet_size)) ** 0.5)
                # height (== width) for the new position embedding
                new_size = int((num_patches // (args.num_frames // self.unimodal_model.patch_embed.tubelet_size)) ** 0.5)
                # class_token and dist_token are kept unchanged
                if orig_size != new_size:
                    print("Position interpolate from %dx%d to %dx%d" % (orig_size, orig_size, new_size, new_size))
                    extra_tokens = pos_embed_checkpoint[:, :num_extra_tokens]
                    # only the position tokens are interpolated
                    pos_tokens = pos_embed_checkpoint[:, num_extra_tokens:]
                    # B, L, C -> BT, H, W, C -> BT, C, H, W
                    pos_tokens = pos_tokens.reshape(-1, args.num_frames // self.unimodal_model.patch_embed.tubelet_size, orig_size,
                                                    orig_size, embedding_size)
                    pos_tokens = pos_tokens.reshape(-1, orig_size, orig_size, embedding_size).permute(0, 3, 1, 2)
                    pos_tokens = torch.nn.functional.interpolate(
                        pos_tokens, size=(new_size, new_size), mode='bicubic', align_corners=False)
                    # BT, C, H, W -> BT, H, W, C ->  B, T, H, W, C
                    pos_tokens = pos_tokens.permute(0, 2, 3, 1).reshape(-1,
                                                                        args.num_frames // self.unimodal_model.patch_embed.tubelet_size,
                                                                        new_size, new_size, embedding_size)
                    pos_tokens = pos_tokens.flatten(1, 3)  # B, L, C
                    new_pos_embed = torch.cat((extra_tokens, pos_tokens), dim=1)
                    checkpoint_model['pos_embed'] = new_pos_embed

            self.utils_load_state_dict(self.unimodal_model, checkpoint_model, prefix=args.model_prefix)

    def utils_load_state_dict(self, model, state_dict, prefix='', ignore_missing="relative_position_index"):
        missing_keys = []
        unexpected_keys = []
        error_msgs = []
        metadata = getattr(state_dict, '_metadata', None)
        state_dict = state_dict.copy()
        if metadata is not None:
            state_dict._metadata = metadata

        def load(module, prefix=''):
            local_metadata = {} if metadata is None else metadata.get(
                prefix[:-1], {})
            module._load_from_state_dict(
                state_dict, prefix, local_metadata, True, missing_keys, unexpected_keys, error_msgs)
            for name, child in module._modules.items():
                if child is not None:
                    load(child, prefix + name + '.')

        load(model, prefix=prefix)

        warn_missing_keys = []
        ignore_missing_keys = []
        for key in missing_keys:
            keep_flag = True
            for ignore_key in ignore_missing.split('|'):
                if ignore_key in key:
                    keep_flag = False
                    break
            if keep_flag:
                warn_missing_keys.append(key)
            else:
                ignore_missing_keys.append(key)

        missing_keys = warn_missing_keys

        if len(missing_keys) > 0:
            print("Weights of {} not initialized from pretrained model: {}".format(
                model.__class__.__name__, missing_keys))
        if len(unexpected_keys) > 0:
            print("Weights from pretrained model not used in {}: {}".format(
                model.__class__.__name__, unexpected_keys))
        if len(ignore_missing_keys) > 0:
            print("Ignored weights of {} not initialized from pretrained model: {}".format(
                model.__class__.__name__, ignore_missing_keys))
        if len(error_msgs) > 0:
            print('\n'.join(error_msgs))


# DISCARD
class VideoEncoder_v2(nn.Module):
    def __init__(self, config):
        super(VideoEncoder_v2, self).__init__()
        self.config = config

        video_mae_config = config['video_mae_configs']
        self.parser = ArgumentParser()
        self.helper_args = self.parser.parse_args()
        with open(video_mae_config['helper_arg_path'], 'r') as f:
            self.helper_args.__dict__ = json.load(f)

        self.unimodal_model = create_model(
            self.helper_args.model,
            img_size=self.helper_args.input_size,
            pretrained=False,
            num_classes=self.helper_args.nb_classes,
            all_frames=self.helper_args.num_frames * self.helper_args.num_segments,
            tubelet_size=self.helper_args.tubelet_size,
            drop_rate=self.helper_args.drop,
            drop_path_rate=self.helper_args.drop_path,
            attn_drop_rate=self.helper_args.attn_drop_rate,
            head_drop_rate=self.helper_args.head_drop_rate,
            drop_block_rate=None,
            use_mean_pooling=self.helper_args.use_mean_pooling,
            init_scale=self.helper_args.init_scale,
            with_cp=self.helper_args.with_checkpoint,
        )
        self.load_and_initiate_unimodal_model()

        self.head = nn.Sequential(
            nn.PReLU(),
            nn.Linear(768, config['common_embedding_dim']),
        )

    def forward(self, x):
        x = self.unimodal_model.forward(x)

        x = self.head(x)
        return x

    def utils_load_state_dict(self, model, state_dict):
        # copy state_dict so _load_from_state_dict can modify it
        metadata = getattr(state_dict, '_metadata', None)
        state_dict = state_dict.copy()
        if metadata is not None:
            state_dict._metadata = metadata

    def load_and_initiate_unimodal_model(self):
        args = self.helper_args

        patch_size = self.unimodal_model.patch_embed.patch_size
        print("Patch size = %s" % str(patch_size))
        args.patch_size = patch_size

        if args.finetune:
            if args.finetune.startswith('https'):
                checkpoint = torch.hub.load_state_dict_from_url(
                    args.finetune, map_location='cpu', check_hash=True)
            else:
                checkpoint = torch.load(args.finetune, map_location='cpu')

            print("Load ckpt from %s" % args.finetune)
            checkpoint_model = None
            for model_key in args.model_key.split('|'):
                if model_key in checkpoint:
                    checkpoint_model = checkpoint[model_key]
                    print("Load state_dict by model_key = %s" % model_key)
                    break
            if checkpoint_model is None:
                checkpoint_model = checkpoint
            for old_key in list(checkpoint_model.keys()):
                if old_key.startswith('_orig_mod.'):
                    new_key = old_key[10:]
                    checkpoint_model[new_key] = checkpoint_model.pop(old_key)

            state_dict = self.unimodal_model.state_dict()
            for k in ['head.weight', 'head.bias']:
                if k in checkpoint_model and checkpoint_model[k].shape != state_dict[k].shape:
                    if checkpoint_model[k].shape[0] == 710 and args.data_set.startswith('Kinetics'):
                        print(f'Convert K710 head to {args.data_set} head')
                        if args.data_set == 'Kinetics-400':
                            label_map_path = 'misc/label_710to400.json'
                        elif args.data_set == 'Kinetics-600':
                            label_map_path = 'misc/label_710to600.json'
                        elif args.data_set == 'Kinetics-700':
                            label_map_path = 'misc/label_710to700.json'

                        label_map = json.load(open(label_map_path))
                        checkpoint_model[k] = checkpoint_model[k][label_map]
                    else:
                        print(f"Removing key {k} from pretrained checkpoint")
                        del checkpoint_model[k]  # JZ: Yes We need this

            all_keys = list(checkpoint_model.keys())
            new_dict = OrderedDict()
            for key in all_keys:
                if key.startswith('backbone.'):
                    new_dict[key[9:]] = checkpoint_model[key]
                elif key.startswith('encoder.'):
                    new_dict[key[8:]] = checkpoint_model[key]
                else:
                    new_dict[key] = checkpoint_model[key]
            checkpoint_model = new_dict

            # interpolate position embedding
            if 'pos_embed' in checkpoint_model:
                pos_embed_checkpoint = checkpoint_model['pos_embed']
                embedding_size = pos_embed_checkpoint.shape[-1]  # channel dim
                num_patches = self.unimodal_model.patch_embed.num_patches  #
                num_extra_tokens = self.unimodal_model.pos_embed.shape[-2] - num_patches  # 0/1

                # height (== width) for the checkpoint position embedding
                orig_size = int(
                    ((pos_embed_checkpoint.shape[-2] - num_extra_tokens) //
                     (args.num_frames // self.unimodal_model.patch_embed.tubelet_size)) ** 0.5)
                # height (== width) for the new position embedding
                new_size = int(
                    (num_patches //
                     (args.num_frames // self.unimodal_model.patch_embed.tubelet_size)) ** 0.5)
                # class_token and dist_token are kept unchanged
                if orig_size != new_size:
                    print("Position interpolate from %dx%d to %dx%d" %
                          (orig_size, orig_size, new_size, new_size))
                    extra_tokens = pos_embed_checkpoint[:, :num_extra_tokens]
                    # only the position tokens are interpolated
                    pos_tokens = pos_embed_checkpoint[:, num_extra_tokens:]
                    # B, L, C -> BT, H, W, C -> BT, C, H, W
                    pos_tokens = pos_tokens.reshape(
                        -1, args.num_frames // self.unimodal_model.patch_embed.tubelet_size,
                        orig_size, orig_size, embedding_size)
                    pos_tokens = pos_tokens.reshape(-1, orig_size, orig_size,
                                                    embedding_size).permute(
                        0, 3, 1, 2)
                    pos_tokens = torch.nn.functional.interpolate(
                        pos_tokens,
                        size=(new_size, new_size),
                        mode='bicubic',
                        align_corners=False)
                    # BT, C, H, W -> BT, H, W, C ->  B, T, H, W, C
                    pos_tokens = pos_tokens.permute(0, 2, 3, 1).reshape(
                        -1, args.num_frames // self.unimodal_model.patch_embed.tubelet_size,
                        new_size, new_size, embedding_size)
                    pos_tokens = pos_tokens.flatten(1, 3)  # B, L, C
                    new_pos_embed = torch.cat((extra_tokens, pos_tokens), dim=1)
                    checkpoint_model['pos_embed'] = new_pos_embed
            elif args.input_size != 224:
                pos_tokens = self.unimodal_model.pos_embed
                org_num_frames = 16
                T = org_num_frames // args.tubelet_size
                P = int((pos_tokens.shape[1] // T) ** 0.5)
                C = pos_tokens.shape[2]
                new_P = args.input_size // patch_size[0]
                # B, L, C -> BT, H, W, C -> BT, C, H, W
                pos_tokens = pos_tokens.reshape(-1, T, P, P, C)
                pos_tokens = pos_tokens.reshape(-1, P, P, C).permute(0, 3, 1, 2)
                pos_tokens = torch.nn.functional.interpolate(
                    pos_tokens,
                    size=(new_P, new_P),
                    mode='bicubic',
                    align_corners=False)
                # BT, C, H, W -> BT, H, W, C ->  B, T, H, W, C
                pos_tokens = pos_tokens.permute(0, 2, 3,
                                                1).reshape(-1, T, new_P, new_P, C)
                pos_tokens = pos_tokens.flatten(1, 3)  # B, L, C
                self.unimodal_model.pos_embed = pos_tokens  # update
            if args.num_frames != 16:
                org_num_frames = 16
                T = org_num_frames // args.tubelet_size
                pos_tokens = self.unimodal_model.pos_embed
                new_T = args.num_frames // args.tubelet_size
                P = int((pos_tokens.shape[1] // T) ** 0.5)
                C = pos_tokens.shape[2]
                pos_tokens = pos_tokens.reshape(-1, T, P, P, C)
                pos_tokens = pos_tokens.permute(0, 2, 3, 4,
                                                1).reshape(-1, C, T)  # BHW,C,T
                pos_tokens = torch.nn.functional.interpolate(
                    pos_tokens, size=new_T, mode='linear')
                pos_tokens = pos_tokens.reshape(1, P, P, C,
                                                new_T).permute(0, 4, 1, 2, 3)
                pos_tokens = pos_tokens.flatten(1, 3)
                self.unimodal_model.pos_embed = pos_tokens  # update

            self.utils_load_state_dict(self.unimodal_model, checkpoint_model)
