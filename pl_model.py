import lightning as L
from simba import Text_Mmamba
from transformers import T5EncoderModel, T5Tokenizer
from torch.optim import AdamW
import torch
from torch import nn
import torch.nn.functional as F
import math
from utils import *

def random_zeroing(batch_tensor, batch_zero_prob=0.1, bit_zero_prob=0.1):
    """
    Args:
    - batch_tensor (torch.Tensor): 输入的0, 1向量，维度为(B, L)
    - batch_zero_prob (float): 每个batch全变成0的概率
    - bit_zero_prob (float): batch中的1变成0的概率
    
    Returns:
    - torch.Tensor: 经过随机置零后的向量
    """
    batch_size, seq_length = batch_tensor.shape

    # Step 1: 每个batch有10%的几率全变成0
    batch_mask = torch.bernoulli(torch.full((batch_size,), 1 - batch_zero_prob))
    # print(batch_mask.bool())
    batch_mask = batch_mask.bool()
    batch_tensor[~batch_mask] = 0
    # print(batch_tensor)

    # Step 2: 对于剩下的batch中的1，以5%的几率变成0
    bit_mask = torch.bernoulli(torch.full_like(batch_tensor.float(), 1 - bit_zero_prob))
    bit_mask = bit_mask.bool()
    # print(bit_mask)
    batch_tensor[batch_mask] = batch_tensor[batch_mask] * bit_mask[batch_mask]

    return batch_tensor

class Text_Mmamba_pl(L.LightningModule):
    def __init__(self, config):
        super().__init__()
        # some setting
        self.config = config
        
        # import the text encoder here
        text_encoder_name = 'google/flan-t5-base'
        self.tokenizer = T5Tokenizer.from_pretrained(text_encoder_name)
        self.text_encoder = T5EncoderModel.from_pretrained(text_encoder_name).train(mode=False)
        self.text_encoder = self.text_encoder.to(self.device)
        
        # import the main LM model
        try:
            self.music_model = Text_Mmamba(
                layers = self.config['model']['layers'],
                codec_layer = self.config['model']['codec_layer'],
                vocab_size = self.config['model']['vocab_size'],
                d_model = self.config['model']['d_model'],
                drop_p = self.config['model']['drop_p'], 
                d_state = self.config['model']['d_state'], 
                num_heads = self.config['model']['num_heads'],
                self_atten_layers = self.config['model']['self_atten_layers'],
                is_incontext = self.config['model']['is_incontext'],
                is_pure_mamba = self.config['model']['is_pure_mamba'],
                )
        except:
            self.music_model = Text_Mmamba(
                layers = self.config['model']['layers'],
                codec_layer = self.config['model']['codec_layer'],
                vocab_size = self.config['model']['vocab_size'],
                d_model = self.config['model']['d_model'],
                drop_p = self.config['model']['drop_p'], 
                d_state = self.config['model']['d_state'],
                )
        
        # log for the configuration
        self.save_hyperparameters()
        self.training_step_outputs = []
        
    def training_step(self, batch, batch_idx):
        # print(batch)
        x, x_mask, y, text = batch
        text_embedding = text
        
        # # process text
        batch_text = self.tokenizer(
            text, padding=True, return_tensors="pt"
        )
        input_ids, attention_mask = batch_text.input_ids.to('cuda'), batch_text.attention_mask.to('cuda')

        
        with torch.no_grad():
            text_embedding = self.text_encoder(
                input_ids=input_ids, attention_mask=attention_mask
            ).last_hidden_state
        
        # dropout texts w/ 10%
        # dropout words w/ 5%
        attention_mask = random_zeroing(attention_mask)
        text_embedding_mask = (attention_mask == 1)
        # text_embedding = text_embedding.to(device)
        
        torch.cuda.set_device(x.device.index)

        output_logit = self.music_model(x, text_embedding, text_embedding_mask)
        
        losses = 0
        for k in range(self.config['model']['codec_layer']):
            logits_k = output_logit[:, k, :, :].contiguous().view(-1, output_logit.size(-1))
            targets_k = y[:, k, :].contiguous().view(-1)
            # logits_mask = mask[:, k, 1:].contiguous().view(-1)
            loss = nn.CrossEntropyLoss(ignore_index=1024, label_smoothing=0.1)(logits_k, targets_k)
            
            losses += loss
        self.training_step_outputs.append( losses / self.config['model']['codec_layer'] )
        return losses
    
    @torch.no_grad()
    def forward(self, description, length=500, temp=1.2, topk=250, g_scale=5, is_flash=False, is_reture_attention=False):
        # append unconditional text
        gen_num = len(description)
        for n in range(gen_num):
            description.append('')
        
        prompt_seq = create_empty_prompt(num=len(description)//2, layer_num=self.config['model']['codec_layer'])
        # print(prompt_seq.shape, self.config['model']['layers'])
        input_seq = torch.LongTensor(prompt_seq).to(self.device)
        
        # process text
        batch_text = self.tokenizer(
            description, padding=True, return_tensors="pt"
        )
        input_ids, attention_mask = batch_text.input_ids.to(self.device), batch_text.attention_mask.to(self.device)
        
        if input_ids.shape[1] < 60:
            pad_text = torch.ones((input_ids.shape[0], 60-input_ids.shape[1]), dtype=torch.int64).to(self.device)
            pad_mask = torch.zeros((input_ids.shape[0], 60-attention_mask.shape[1]), dtype=torch.int64).to(self.device)
            
            input_ids = torch.cat([input_ids, pad_text], dim=1)
            attention_mask = torch.cat([attention_mask, pad_mask], dim=1)

        with torch.set_grad_enabled(False):
            text_embedding = self.text_encoder(
                input_ids=input_ids, attention_mask=attention_mask
            ).last_hidden_state
        
        # print(text_embedding.shape)
        text_embedding_mask = (attention_mask == 1).to(self.device)
        torch.cuda.set_device(self.device.index)
        B, K, L = prompt_seq.shape
        # print(B, K, L)
        
        # while L < length+10:
        from tqdm import tqdm
        for gen in tqdm(range(length+10)):
            # print(L)
            cond_output_logits = self.music_model(torch.LongTensor(prompt_seq).to(self.device), text_embedding[:len(description)//2], text_embedding_mask[:len(description)//2])
            uncond_output_logits = self.music_model(torch.LongTensor(prompt_seq).to(self.device), text_embedding[len(description)//2:], text_embedding_mask[len(description)//2:])

            output_logits = uncond_output_logits + (cond_output_logits - uncond_output_logits) * g_scale

            _logit = output_logits[:, :, -1, :].to('cpu').detach().numpy()

            batch_new = []
            for b in range(B):
                words = []
                for i in range(K):
                    word = temperature_sampling(
                            logits=_logit[b, i],
                            temperature=temp,
                            topk=topk)
                    words.append([word])
                batch_new.append(words)
            prompt_seq = np.concatenate((prompt_seq, batch_new), axis=2)
            # L+=1
        
        prompt_seq = to_parallel(prompt_seq)[:, :, :length]
        return prompt_seq
    
    def on_before_optimizer_step(self, optimizer):
        step_mean = torch.stack(self.training_step_outputs).mean()
        self.log('train_loss_steps', step_mean)
        self.training_step_outputs.clear()
        
    
    def configure_optimizers(self):
        # optimizer setting
        optimizer = AdamW(  self.music_model.parameters(), 
                            lr = self.config['optimizer']['optim_lr'], 
                            weight_decay = self.config['optimizer']['weight_decay'], 
                            betas = self.config['optimizer']['betas'],
                            eps = 1e-07)
        warm_up_iter = 500 
        T_max = self.trainer.estimated_stepping_batches+1
        
        lr_max = 1e-1	                        # 最大值
        lr_min = 5e-4	                        # 最小值

        lambda0 = lambda cur_iter: cur_iter / warm_up_iter if  cur_iter < warm_up_iter else \
        (lr_min + 0.5*(lr_max-lr_min)*(1.0+math.cos( (cur_iter-warm_up_iter)/(T_max-warm_up_iter)*math.pi)))/0.1
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda0)
        scheduler.step()
        
        return [optimizer], [scheduler]