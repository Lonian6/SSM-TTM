import numpy as np
from torch import nn
import torch

SPECIAL_ID=1024

# code pattern
def to_delay(para):
    '''
    INPUT:
    tokens: a k layers RVQ token
    1 5 9... (course layer)
    2 6 10...
    3 7 11...
    4 8 12... (fine layer)
    
    OUTPUT:
    delay: the delay pattern
    p 1 5 9... (course layer)
    p p 2 6 10...
    p p p 3 7 11...
    p p p p 4 8 12... (fine layer)
    '''
    if len(para.shape) == 2:
        K, L = para.shape
        delay = np.zeros((K, L)) + SPECIAL_ID
        for i in range(K):
            # delay[i, i:] = para[i, :500-i]
            delay[i, i+1:] = para[i, :L-1-i]
    elif len(para.shape) == 3:
        B, K, L = para.shape
        delay = np.zeros((B, K, L)) + SPECIAL_ID
        for i in range(K):
            delay[:, i, i+1:] = para[:, i, :L-1-i]
    return delay

def to_parallel(delay):
    '''
    INPUT:
    delay: the delay pattern
    p 1 5 9... (course layer)
    p p 2 6 10...
    p p p 3 7 11...
    p p p p 4 8 12... (fine layer)
    
    
    OUTPUT:
    para: a k layers RVQ token w/ parallel form
    1 5 9... (course layer)
    2 6 10...
    3 7 11...
    4 8 12... (fine layer)
    '''
    if len(delay.shape)==2:
        K, L = delay.shape
        para = np.zeros((K, L)) + SPECIAL_ID
        for i in range(K):
            # para[i, :500-i] = delay[i, i:]
            para[i, :L-1-i] = delay[i, i+1:]
    elif len(delay.shape)==3:
        B, K, L = delay.shape
        para = np.zeros((B, K, L)) + SPECIAL_ID
        for i in range(K):
            # para[i, :500-i] = delay[i, i:]
            para[:, i, :L-1-i] = delay[:, i, i+1:]
    return para

def create_empty_prompt(num, layer_num=1, SPECIAL_ID=SPECIAL_ID):
    '''
    INPUT:
    num: batch number
    
    OUTPUT:
    empty tensor w/ [batch, 4, 1] shape
    '''
    prompt = np.zeros((num, layer_num, 1)) + SPECIAL_ID
    return prompt

# Sampling
def temperature_sampling(logits, temperature, topk):
    # probs = np.exp(logits / temperature) / np.sum(np.exp(logits / temperature))
    logits = torch.Tensor(logits)
    probs = nn.Softmax(dim=0)(logits / temperature)
    probs = np.array(probs)
    if topk == 1:
        prediction = np.argmax(probs)
    else:
        sorted_index = np.argsort(probs)[::-1]
        candi_index = sorted_index[:topk]
        candi_probs = [probs[i] for i in candi_index]
        # normalize probs
        candi_probs /= sum(candi_probs)
        # choose by predicted probs
        prediction = np.random.choice(candi_index, size=1, p=candi_probs)[0]
    return prediction

if __name__ == '__main__':
    # test code pattern
    np.set_printoptions(suppress=True)
    a = np.random.randint(5, size=(4, 6))
    print(a)
    delay_a = to_delay(a)
    print(delay_a)
    para_a = to_parallel(delay_a)
    print(para_a)
    print('Done')
    
    # test