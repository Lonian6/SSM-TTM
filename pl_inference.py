import sys
import torch
from torch.utils.data import Dataset, DataLoader
from torch import nn
import torch.nn.functional as F
from pl_model import Text_Mmamba_pl
from glob import glob
import numpy as np
import os
import json
from tqdm import tqdm
import math
# import argparse
from transformers import T5EncoderModel, T5Tokenizer
# from text_simba import MB_Dataset
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
os.environ["TOKENIZERS_PARALLELISM"] = "false"
# torch.multiprocessing.set_start_method('spawn')
from utils import *
import argparse

def create_logger(logger_file_path, name=None):
    import time
    import logging
    
    if not os.path.exists(logger_file_path):
        os.makedirs(logger_file_path)
    if name is not None:
        log_name = '{}.log'.format(time.strftime('%Y-%m-%d-%H-%M'))
    else:
        log_name = '{}.log'.format(name)
    final_log_file = os.path.join(logger_file_path, log_name)

    logger = logging.getLogger()  # 设定日志对象
    logger.setLevel(logging.INFO)  # 设定日志等级

    file_handler = logging.FileHandler(final_log_file)  # 文件输出
    console_handler = logging.StreamHandler()  # 控制台输出

    # 输出格式
    formatter = logging.Formatter(
        "%(asctime)s %(levelname)s: %(message)s "
    )

    file_handler.setFormatter(formatter)  # 设置文件输出格式
    console_handler.setFormatter(formatter)  # 设施控制台输出格式
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    return logger

def parse_opt():
    parser = argparse.ArgumentParser()
    # general
    parser.add_argument('--model_path', type=str,
                        help='path of model ckpt', default='./ckpt/simba.ckpt')
    parser.add_argument('--save_dir_name', type=str,
                        help='path to save outputs', default='test_outputs')
    parser.add_argument('--config_path', type=str,
                        help='path of model config', default='./ckpt/config.json')
    
    args = parser.parse_args()
    return args

opt = parse_opt()

with open(opt.config_path) as f:
    config = json.load(f)
model = Text_Mmamba_pl.load_from_checkpoint(opt.model_path, config)
model.eval()
model.freeze()
# folder_name = 'musicgen_baseline'
save_path = f'./outputs/{opt.save_dir_name}/dac_token'
os.makedirs(save_path, exist_ok=True)

captions = [
    "A bluesy acoustic guitar solo with expressive slides and fingerpicked melodies. The warm tone of the guitar is complemented by a subtle shuffle rhythm in the background, creating a nostalgic and intimate mood. This piece would suit a road trip scene or a cozy evening by the fireplace.",
    "A deep house track with a hypnotic bassline and a steady kick-clap rhythm. The synth pads create a lush and immersive atmosphere, while subtle effects add movement. This could be playing in a high-end lounge or during a late-night DJ set.",
    "A reggae instrumental with a relaxed groove, featuring offbeat electric guitar skanks, a deep, laid-back bassline, and a gentle horn section. The rhythm section provides a smooth bounce, creating a sunny and carefree vibe. This track would be perfect for a tropical island setting or a beachside café.",
    "A funky jazz-fusion groove featuring a slap bass riff, syncopated electric piano chords, and lively brass stabs. The drums maintain a tight rhythm with ghost notes on the snare, keeping the energy high. This piece would suit a late-night urban scene or a stylish heist film.",
    "A heavy rock instrumental driven by distorted electric guitars and a powerful drum groove. The bassline adds weight to the mix, while an energetic guitar solo shreds in the forefront. This song would be fitting for an intense action sequence or a high-adrenaline sports montage.",
    "A melancholic solo piano piece with delicate, flowing melodies and soft pedal resonance. The tempo is slow, and the dynamics shift gently, evoking introspection and deep emotion. This could be playing in the background of a heartfelt movie scene or during a quiet, rainy evening.",
    "A high-energy electronic dance track with a pulsating four-on-the-floor kick drum and shimmering hi-hats. Deep synth bass drives the rhythm, while bright plucks and airy pads create an uplifting mood. This track would fit well in a festival setting or a high-speed car chase in a video game.",
    "A dreamy ambient track featuring a soft pad melody and gentle synth arpeggios. A warm bassline adds depth while distant chimes and subtle white noise create an atmospheric, floating sensation. This piece could be playing in a meditation session or a futuristic sci-fi scene.",
    "A soft acoustic guitar gently strums a calming melody, accompanied by light percussion. A warm, deep bass softly hums in the background. The atmosphere is peaceful and relaxing, making it perfect for a quiet morning or a cozy café.",
    "A smooth jazz piece led by a gentle piano melody, accompanied by a double bass and soft brush drumming. A saxophone occasionally plays a few warm notes, adding to the relaxed atmosphere. This could be the perfect background music for a late-night coffee shop.",
    "A lively funk groove with a punchy bassline and rhythmic electric guitar strumming. The drums keep a steady beat, while a brass section occasionally joins in with energetic stabs. This track could be playing at a retro dance party.",
    "A high-energy rock track with distorted electric guitars and a driving drum beat. The rhythm is fast and powerful, with occasional breaks that add intensity. It feels like a song that could be playing in a road trip montage or an action-packed scene.",
    "This song seems to be experimental. An e-bass is playing a light funky groove along with a clapping sound while other percussion's are adding a rhythm rounding up the rhythm section. A oriental sounding string is playing one single note rhythmically while another oriental and plucked instrument is playing a melody in the higher register. The whole recording sounds a little old but is of good quality. This song may be playing in the kitchen while cooking.",
    "A groovy house track featuring punchy kick drums, smooth basslines, and uplifting synth chords. The steady four-on-the-floor beat keeps the momentum, while shimmering pads and subtle FX create a hypnotic dancefloor vibe.",
    "A high-energy rock instrumental driven by electrifying guitar riffs, powerful drum beats, and dynamic bass grooves. The track builds with intensity, featuring soaring solos and a raw, rebellious spirit that ignites adrenaline.",
    "A fast-paced electronic track with pounding kick drums, rolling basslines, and evolving synth sequences. The relentless groove, layered with atmospheric effects, builds tension and energy, perfect for a club setting."
]

with open(f'./outputs/{opt.save_dir_name}/music_captions.json', "w", encoding="utf-8") as f:
    json.dump(captions, f, ensure_ascii=False, indent=4)
L = 2200
# L=480
audio_num = 0
# with torch.autocast(device_type="cuda", dtype=torch.float32):
with torch.autocast(device_type="cuda", dtype=torch.float32):
    with torch.no_grad():
        device = 'cuda'
        for idx, i in enumerate(captions):
            print(f'Processing caption {idx}: {i}', end='\r')
            for _ in range(2):
                if os.path.isfile(os.path.join(save_path, f'{idx}_{audio_num}.npy')):
                    audio_num += 5
                    print(f'{idx}_{audio_num}.npy')
                    continue
                description = [i] * 5
                # print(len(description), len(i))
                # break
                # print(len(i['ytid']))
                prompt_seq = model(description=description, length=L, g_scale=3)
                # print(prompt_seq.shape, len(description))

                for b in range(5):
                    np.save(os.path.join(save_path, f'{idx}_{audio_num}.npy'), prompt_seq[b, :, :L])
                    audio_num += 1
