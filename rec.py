from inference import recItem
import os
os.environ["CUDA_VISIBLE_DEVICES"]="MIG-b0dc7640-8329-5aa9-a5eb-71c8d36164b1"  # specify which GPU(s) to be used
import time
import torch
import argparse
import sys
import pickle

from model import SASRec
from utils import *
from torch.utils.tensorboard import SummaryWriter
writer = SummaryWriter()

def str2bool(s):
    if s not in {'false', 'true'}:
        raise ValueError('Not a valid boolean string')
    return s == 'true'

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', default='AmazonBooks', type=str)
parser.add_argument('--train_dir', default='default', type=str)
parser.add_argument('--batch_size', default=256, type=int)
parser.add_argument('--lr', default=0.00001, type=float)
parser.add_argument('--maxlen', default=200, type=int)
parser.add_argument('--hidden_units', default=50, type=int)
parser.add_argument('--num_blocks', default=2, type=int)
parser.add_argument('--num_epochs', default=200, type=int)
parser.add_argument('--num_heads', default=1, type=int)
parser.add_argument('--dropout_rate', default=0.5, type=float)
parser.add_argument('--l2_emb', default=0.0, type=float)
parser.add_argument('--device', default='cuda', type=str)
parser.add_argument('--inference_only', default=False, type=str2bool)
parser.add_argument('--state_dict_path', default="AmazonBooks_default/Amazon_SASRec.epoch=200.lr=0.001.layer=2.head=1.hidden=50.maxlen=200.pth", type=str)

args = parser.parse_args()

dataset = data_partition(args.dataset)

[user_train, user_valid, user_test, usernum, itemnum] = dataset

user_train = sorted(user_train.values(), key=lambda x:len(x), reverse=True)

model = SASRec(usernum, itemnum, args).to(args.device)

model.load_state_dict(torch.load(args.state_dict_path, map_location=torch.device(args.device)))

item_idx = [111, 200, 3, 400, 112]

k = 5

recItem(model, user_train, item_idx, k, args)

