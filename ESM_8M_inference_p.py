import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from pathlib import Path

import numpy as np
import time
import datetime
import random
import os
from collections import Counter
import torch.nn.functional as F
import pandas as pd

import torchvision.transforms as transforms
from models import ProteinClassifier
from train_func import train_one_epoch, test_one_epoch

from transformers import AutoTokenizer, EsmModel

import copy

import argparse


## pars
def get_args_parser():
    parser = argparse.ArgumentParser('Set BKD', add_help=False)
    parser.add_argument('--lr', default=1e-4, type=float,
                        help='learning rate')

    parser.add_argument('--lr_drop', default=0.1, type=float)  
    parser.add_argument('--epoch', default=100, type=int)  
    parser.add_argument('--KD_epoch', default=120, type=int)  
    parser.add_argument('--num_classes', default=10, type=int)
    parser.add_argument('--hidden_size', default=128, type=int)
    parser.add_argument('--alpha',default=0.3, type=float)
    parser.add_argument('--epsilon_1',default=1e-1, type=float)
    parser.add_argument('--epsilon_2',default=1e-3, type=float)
    parser.add_argument('--T',default=8, type=int,
                        help='temperature of KD loss')
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--batch_size', default=32, type=int)
    parser.add_argument('--num_workers', default=2, type=int)
    parser.add_argument('--device', default='cuda')
    
    parser.add_argument('--path_plot',default='./plot_student/')
    parser.add_argument('--LMC_params',default='./LMC_params/')
    parser.add_argument('--testdata_dir',default="./data/data_our_test.csv")
    parser.add_argument('--pred_dir',default='./result_pred_p')

    return parser
  
  

def main(args):
  seed = args.seed
  torch.manual_seed(seed)
  np.random.seed(seed)
  random.seed(seed)

  device = torch.device(args.device)
  
  # Get start time
  current_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
  print("++++++++++++++++++++++++++")
  print("start time: ", current_time)
  
  ##------------------Prepare dataset---------
  tokenizer = AutoTokenizer.from_pretrained("facebook/esm2_t6_8M_UR50D")
  
  df_test = pd.read_csv(args.testdata_dir)
  test_data = ProteinDataset(df_test, tokenizer)
  test_loader = DataLoader(test_data, batch_size=args.batch_size, shuffle=False)

  
  pred_dir = args.pred_dir
  os.makedirs(pred_dir, exist_ok=True)
  
  ##
  esm_model = EsmModel.from_pretrained("facebook/esm2_t6_8M_UR50D")
  model_infer = ProteinClassifier(esm_model, args.hidden_size, args.num_classes).to(device)
  
  
  print("----student model structure:")
  print(model_infer)
  
  # Run the training loop
  random.seed(2024)
  
  pred_p_list_all = []

  print('--------Training starts!')
  start_time=time.time()

  for epoch in range(0,args.epoch):

    if epoch>=args.KD_epoch:
      ###--- 
      pars = torch.load(args.LMC_params+"LMC_params_epoch"+str(epoch)+".pth")
      idx_random = random.sample(range(len(pars)), int(0.2*len(pars))) # sample
      
      ###--- (2) calculate p
      pred_p_list_ep = get_infer_p(test_loader=test_loader, model_params_ep=pars, model_student=model_infer, device=device, idx_random=idx_random)
      pred_p_list_all.append(pred_p_list_ep)
      
      print('EPOCH:%d  Inference Used Time:%s'
            %(epoch, str(datetime.timedelta(seconds=int(time.time()-start_time)))))
      
  #-- save infered p
  torch.save(pred_p_list_all, pred_dir+'/pred_p_list_all_epoch'+str(args.epoch)+'.pth')
    
  #
  print('------Training ends!')





##---- functions --------------------------------------------------
# Define the dataset
class ProteinDataset(Dataset):
    def __init__(self, dataframe, tokenizer):
        self.sequences = dataframe['Sequence'].values
        self.labels = dataframe['Label'].values
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        sequence = self.sequences[idx]
        label = self.labels[idx]
        inputs = self.tokenizer(
            sequence,
            return_tensors="pt",
            padding="max_length",  # Pad to the max length in the tokenizer
            truncation=True,       # Truncate sequences that exceed the max length
            max_length=1000        # Specify a maximum sequence length (adjust based on your model)
        )
        return inputs, torch.tensor(label, dtype=torch.long)



def get_infer_p(test_loader, model_params_ep, model_student, device, idx_random):
  pred_p_list_ep = []
  for itr in idx_random:
    model_params = model_params_ep[itr]
    #update model params:
    model_student.load_state_dict(model_params)
        
    # evaluate:
    model_student.eval()
    with torch.no_grad():
      pred_p = np.empty([0,10])
      # Iterate over the DataLoader for testing data
      for i, data in enumerate(test_loader, 0): 
        # Get and prepare inputs
        inputs, labels = data
        inputs = {key: value.squeeze(0).to(device) for key, value in inputs.items()}
        labels = labels.to(device)

        # Perform forward pass
        outputs = model_student(inputs)
        pred = outputs.detach().cpu().numpy()
        # print("logits:", pred)
        tmp = np.exp(pred-np.max(pred,axis=1,keepdims=True))
        pred = tmp/tmp.sum(axis=1,keepdims=True)
        # print("probability:", pred)
        # pred = pred/pred.sum(axis=1,keepdims=True)
        pred_save = copy.deepcopy(pred)
        pred_p = np.vstack((pred_p,pred_save))
    pred_p_list_ep.append(pred_p)
    # pred_p_all = np.hstack((pred_p_all,pred_p))
  return pred_p_list_ep




#-------------------------------
if __name__ == '__main__':
  parser = argparse.ArgumentParser('BKD training and evaluation script', parents=[get_args_parser()])
  args = parser.parse_args()
  # if args.output_dir:
  #     Path(args.output_dir).mkdir(parents=True, exist_ok=True)
  main(args)


