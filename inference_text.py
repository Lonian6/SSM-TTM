import sys

mode = sys.argv[1]
print(mode)
# pytorch
import torch
from torch.utils.data import Dataset, DataLoader
from torch import nn
import torch.nn.functional as F
from pl_model import Text_Mmamba_pl
# import lightning as L
# from lightning.pytorch.callbacks.early_stopping import EarlyStopping
# others
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

# mode = 'incontext_simba'
# print(mode)

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

if mode == 'incontext_mamba': 
    # check!
    with open('/mnt/gestalt/home/lonian/mamba/model/ckpts/mamba_1_incontext_smalldata_bf16_1129/config.json') as f:
        config = json.load(f)
    model = Text_Mmamba_pl.load_from_checkpoint("/mnt/gestalt/home/lonian/mamba/model/ckpts/mamba_1_incontext_smalldata_bf16_1129/lightning_logs/version_1/checkpoints/epoch=50-step=21500.ckpt", config)

elif mode == 'incontext_simba': 
    # check!
    with open('/mnt/gestalt/home/lonian/mamba/model/ckpts/simba_1_incontext_smalldata_bf16/config.json') as f:
        config = json.load(f)
    # model = Text_Mmamba_pl.load_from_checkpoint("/mnt/gestalt/home/lonian/mamba/model/ckpts/simba_1_incontext_smalldata_bf16/lightning_logs/version_3/checkpoints/epoch=50-step=21500.ckpt", config)
    model = Text_Mmamba_pl.load_from_checkpoint("/mnt/gestalt/home/lonian/mamba/model/ckpts/simba_1_incontext_smalldata_bf16/lightning_logs/version_5/checkpoints/epoch=54-step=23500.ckpt", config)
    
elif mode == 'incontext_transformer':
    with open('/mnt/gestalt/home/lonian/mamba/model/ckpts/victor/to_wei_Transformer_incontext_50epoch/config.json') as f:
        config = json.load(f)
    model = Text_Mmamba_pl.load_from_checkpoint("/mnt/gestalt/home/lonian/mamba/model/ckpts/victor/to_wei_Transformer_incontext_50epoch/epoch=50-step=21500.ckpt", config)

elif mode == 'cross_simba':
    with open('/mnt/gestalt/home/lonian/mamba/model/ckpts/simba_1_cross_smalldata_bf16/config.json') as f:
        config = json.load(f)
    model = Text_Mmamba_pl.load_from_checkpoint("/mnt/gestalt/home/lonian/mamba/model/ckpts/simba_1_cross_smalldata_bf16/lightning_logs/version_2/checkpoints/epoch=50-step=21500.ckpt", config)

elif mode == 'cross_transformer':
    with open('/mnt/gestalt/home/lonian/mamba/model/ckpts/victor/transformer_1_cross_50epoch/config.json') as f:
        config = json.load(f)
    model = Text_Mmamba_pl.load_from_checkpoint("/mnt/gestalt/home/lonian/mamba/model/ckpts/victor/transformer_1_cross_50epoch/epoch=50-step=21500.ckpt", config)

elif mode == 'proposed':
    with open('/mnt/gestalt/home/lonian/mamba/model/ckpts/proposed_v1/config.json') as f:
        config = json.load(f)
    model = Text_Mmamba_pl.load_from_checkpoint("/mnt/gestalt/home/lonian/mamba/model/ckpts/proposed_v1/lightning_logs/version_3/checkpoints/epoch=50-step=21500.ckpt", config)

elif mode == 'proposed_v2':
    with open('/mnt/gestalt/home/lonian/mamba/model/ckpts/proposed_v2_24layers/config.json') as f:
        config = json.load(f)
    model = Text_Mmamba_pl.load_from_checkpoint("/mnt/gestalt/home/lonian/mamba/model/ckpts/proposed_v2_24layers/lightning_logs/version_0/checkpoints/epoch=67-step=29000.ckpt", config)

    
else:
    # with open('/mnt/gestalt/home/lonian/mamba/model/ckpts/dac_1_mamba_cross/config.json') as f:
    #     config = json.load(f)
    # model = Text_Mmamba_pl.load_from_checkpoint("/mnt/gestalt/home/lonian/mamba/model/ckpts/dac_1_mamba_cross/lightning_logs/version_0/checkpoints/epoch=121-step=50000.ckpt", config)
    with open('/mnt/gestalt/home/lonian/mamba/model/ckpts/simba_1_cross_smalldata_bf16/config.json') as f:
        config = json.load(f)
    model = Text_Mmamba_pl.load_from_checkpoint("/mnt/gestalt/home/lonian/mamba/model/ckpts/simba_1_cross_smalldata_bf16/lightning_logs/version_2/checkpoints/epoch=54-step=23500.ckpt", config)
    # save_path = f'/mnt/gestalt/home/lonian/mamba/test_results/transformer_1_cross_smalldata_bf16_10000steps'



# with open('/mnt/gestalt/home/lonian/datasets/MusicBench/MusicBench_test_A_simba.json') as f:
#     meta_A = json.load(f)
# import datasets
# datasets_200 = datasets.load_from_disk('/mnt/gestalt/home/lonian/datasets/MusicCaps/eval_sub_200')
# loader = DataLoader(dataset=datasets_200, batch_size = 10)

# model = Text_Mmamba_pl.load_from_checkpoint("/mnt/gestalt/home/lonian/mamba/model/ckpts/dac_1_transformer/lightning_logs/version_3/checkpoints/epoch=61-step=25500.ckpt", config)

model.eval()
model.freeze()
folder_name = 'description'
save_path = f'/mnt/gestalt/home/lonian/mamba/exp_results/{folder_name}/{mode}/dac_token'
os.makedirs (save_path, exist_ok=True)

logger = create_logger(f'/mnt/gestalt/home/lonian/mamba/exp_results/{folder_name}/{mode}', name=mode)

logger.info(f'Is incontext: {config['model']['is_incontext']}')
logger.info(f'Attention layers: {config['model']['self_atten_layers']}')
logger.info(f'Is pure mamba: {config['model']['is_pure_mamba']}')
# "self_atten_layers": [],
#         "is_incontext": true,
#         "is_pure_mamba": false
L = 2588//3
with torch.no_grad():
    device = 'cuda'
    # for i in tqdm(loader):
    #     if os.path.isfile(os.path.join(save_path, '{}.npy'.format(i['ytid'][0]))):
    #         continue
    description = ['Pop music', 'Jazz', 'Electronic Dance Music', 'Rock music with heavy metal style']
    # print(len(i['ytid']))
    prompt_seq = model(description=description, length=L, g_scale=3)
    # print(prompt_seq.shape)

    for b in range(len(description)):
        np.save(os.path.join(save_path, '{}.npy'.format(b)), prompt_seq[b, :, :L])
        
        # break

    # with open(os.path.join(save_path, 'descriptions.json'), 'w', encoding='utf-8') as f:
    #     json.dump(all_des, f, ensure_ascii=False, indent=4)