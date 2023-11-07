import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
from transformers import T5Tokenizer, T5ForConditionalGeneration
from transformers import BertModel, BertTokenizer
import torch
import re
import numpy as np
from Ampep.toxic_func import toxic_feature
from Ampep.amp_func import amp_feature
import random
from MAE_Modules.utils import *

INFER_BATCH_SIZE = 32


tokenizer = T5Tokenizer.from_pretrained('prot_t5_xl_bfd/')
model = T5ForConditionalGeneration.from_pretrained('prot_t5_xl_bfd/').to('cuda')

with open('input.txt', 'r') as f:
    seq = f.readline()
optimize(model,seq)