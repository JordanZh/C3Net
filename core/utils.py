import os
import torchaudio
import torch
import skimage.io as io

from core.common.utils import regularize_image
from core.models.model_module_infer import model_module
from core.models.encoders.ast import AstEncoder

def load_audio_as_fbank(path, as_dummy=False):
    target_length = 1024
    melbins = 128
    norm_mean = -4.2677393
    norm_std = 4.5689974

    waveform, sample_rate = torchaudio.load(path)
    if as_dummy:
        waveform = torch.zeros_like(waveform)
    waveform = waveform - waveform.mean()
    fbank = torchaudio.compliance.kaldi.fbank(waveform, htk_compat=True, sample_frequency=sample_rate, use_energy=False,
                                                  window_type='hanning', num_mel_bins=melbins, dither=0.0, frame_shift=10)
    n_frames = fbank.shape[0]
    p = target_length - n_frames
    if p > 0:
        m = torch.nn.ZeroPad2d((0, 0, 0, p))
        fbank = m(fbank)
    elif p < 0:
        fbank = fbank[0:target_length, :]
    fbank = (fbank - norm_mean) / (norm_std * 2)
    return fbank

def audio_image2text_inference(
        inference_tester,
        save_path=None,
        audio_path=None,
        image_path=None,
        fbank=None,
        img=None
):
    if fbank is None:
        fbank = load_audio_as_fbank(audio_path)
        fbank = fbank.to('cuda')
    if img is None:
        img = io.imread(image_path)
        img = regularize_image(img, image_size=512, mode='test').cuda()

    text = inference_tester.inference_(
        xtype = ['text'],
        condition = [fbank, img],
        condition_types = ['audio', 'image'],
        ddim_steps = 100,
        scale = 7.5,)
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        with open(save_path, 'w') as f:
            f.write(text[0][0])
    return text[0][0]

def audio_text2image_inference(
        inference_tester,
        save_path=None,
        audio_path=None,
        text=None,
        fbank=None
):
    if fbank is None:
        fbank = load_audio_as_fbank(audio_path)
        fbank = fbank.to('cuda')
    
    img = inference_tester.inference_(
        xtype = ['image'],
        condition = [fbank, text],
        condition_types = ['audio', 'text'],
        n_samples = 1,
        scale = 7.5,
        image_size = 512,
        ddim_steps = 50,)[0][0]
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        img.save(save_path)
    return img

def image_text2audio_inference(
        inference_tester,
        save_path=None,
        image_path=None,
        text=None,
        img=None
):
    if img is None:
        img = io.imread(image_path)
        img = regularize_image(img, image_size=512, mode='test').cuda()
    
    audio = inference_tester.inference_(
        xtype = ['audio'],
        condition = [img, text],
        condition_types = ['image', 'text'],
        n_samples = 1,
        scale = 7.5,
        ddim_steps=50,)[0][0]

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        torchaudio.save(save_path, torch.tensor(audio), 16000)
    return audio

def get_inference_model():
    model_load_paths = ['CoDi_encoders.pth', 'CoDi_text_diffuser.pth', 'CoDi_audio_diffuser_m.pth', 'CoDi_video_diffuser_8frames.pth']
    inference_tester = model_module(data_dir='checkpoints/', pth=model_load_paths, fp16=False) # turn on fp16=True if loading fp16 weights
    all_state_dicts = torch.load('checkpoints/Control_models.pth', map_location='cpu')
    inference_tester.net.model.diffusion_model.control_unet_audio.load_state_dict(all_state_dicts['control_audio_model'], strict=True)
    inference_tester.net.model.diffusion_model.control_unet_image.load_state_dict(all_state_dicts['control_image_model'], strict=True)
    inference_tester.net.model.diffusion_model.control_unet_text.load_state_dict(all_state_dicts['control_text_model'], strict=True)
    inference_tester = inference_tester.cuda()
    inference_tester = inference_tester.eval()
    audio_encoder = AstEncoder(pretrained_mdl_path='checkpoints/Control_base.pth')
    audio_encoder.load_state_dict(all_state_dicts['audio_encoder_state_dict'], strict=True)
    audio_encoder = audio_encoder.to('cuda')
    audio_encoder = audio_encoder.eval()
    inference_tester.audio_encoder = audio_encoder
    inference_tester.dummy_fbank = all_state_dicts['dm'].to('cuda')
    return inference_tester
