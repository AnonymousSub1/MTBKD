import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset, Subset
from sklearn.datasets import load_boston
from sklearn.preprocessing import StandardScaler
from pathlib import Path

import numpy as np
from scipy.stats import wishart
import time
import datetime
import random
import matplotlib.pyplot as plt
import os
from collections import Counter
import torch.nn.functional as F

# from PIL import Image
# import torchvision.transforms as transforms
# from torchvision.models import resnet50
# from torchvision.models import ResNet50_Weights

import torch.optim as optim
import torchvision
# import scheduler
import pandas as pd

from models import ProteinClassifier
from train_func import train_one_epoch_with_KD_mix, test_one_epoch_with_KD_mix
from sklearn.model_selection import train_test_split

from collections import Counter
from transformers import AutoTokenizer, EsmModel
from torchvision import datasets, transforms

import copy

import argparse


## pars
def get_args_parser():
    parser = argparse.ArgumentParser('Set BKD', add_help=False)
    parser.add_argument('--lr', default=3e-4, type=float,
                        help='learning rate')
    parser.add_argument('--lr_drop', default=0.1, type=float)  
    parser.add_argument('--epoch', default=300, type=int)  
    parser.add_argument('--KD_epoch', default=120, type=int)  
    parser.add_argument('--num_classes', default=10, type=int)
    parser.add_argument('--hidden_size', default=128, type=int)
    parser.add_argument('--alpha',default=10, type=float)
    parser.add_argument('--epsilon_1',default=8e-8, type=float)
    parser.add_argument('--epsilon_2',default=4e-4, type=float)
    parser.add_argument('--T',default=2, type=int,
                        help='temperature of KD loss')
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--batch_size', default=32, type=int)
    parser.add_argument('--num_workers', default=2, type=int)
    parser.add_argument('--device', default='cuda')
    
    parser.add_argument('--path_plot',default='./plot_student/')
    parser.add_argument('--student_dir',default='./student_model_distill/')
    parser.add_argument('--LMC_params',default='./LMC_params_mix_weighted/')
    parser.add_argument('--traindata_dir',default="./Datasets/deeploc/data_our/data_our_train.csv")
    parser.add_argument('--testdata_dir',default="./Datasets/deeploc/data_our/data_our_test.csv")
    
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
  
  ## Prepare dataset
  tokenizer = AutoTokenizer.from_pretrained("facebook/esm2_t6_8M_UR50D")
  
  df = pd.read_csv(args.traindata_dir)
  teacher_probs1 = pd.read_csv('./teacher_pred_prob/ESM_650M_probs_sample.csv')  # CSV with predicted probabilities (columns: class probabilities)
  teacher_probs2 = pd.read_csv('./teacher_pred_prob/T5_probs_sample.csv')
  # Ensure the probabilities align with the original data
  print("len(df): ", len(df))
  print("len(teacher_probs1): ", len(teacher_probs1))
  print("len(teacher_probs2): ", len(teacher_probs2))
  assert len(df) == len(teacher_probs1) == len(teacher_probs2), "Mismatch in data and teacher probabilities!"

  df_train, df_val = train_test_split(df, test_size=0.2, random_state=42)  # Adjust test_size as needed
  train_data = ProteinDataset_KD(df_train, tokenizer, teacher_probs1=teacher_probs1.loc[df_train.index],
                            teacher_probs2=teacher_probs2.loc[df_train.index])
  val_data = ProteinDataset_KD(df_val, tokenizer, teacher_probs1=teacher_probs1.loc[df_val.index],
                            teacher_probs2=teacher_probs2.loc[df_val.index])
  train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True)
  val_loader = DataLoader(val_data, batch_size=args.batch_size, shuffle=False)

  df_test = pd.read_csv(args.testdata_dir)
  teacher_probs1_test = pd.read_csv('./teacher_pred_prob/ESM_650M_probs_test.csv', header=None)  # CSV with predicted probabilities (columns: class probabilities)
  teacher_probs2_test = pd.read_csv('./teacher_pred_prob/T5_probs_test.csv', header=None)
  test_data = ProteinDataset_KD(df_test, tokenizer, teacher_probs1_test, teacher_probs2_test)
  test_loader = DataLoader(test_data, batch_size=args.batch_size, shuffle=False)
  
  print("train data size: ", df_train.shape)
  print("val data size: ", df_val.shape)
  print("test data size: ", df_test.shape)
  
  NN = len(train_data)
  
  ##
  print('--------------Distill the student model, using original KD!')
  # teacher_model = torch.load(args.pretrained_teacher_dir)
  # teacher_model_T2 = torch.load(args.pretrained_teacher_dir_T2)
  # teacher_model_T3 = torch.load(args.pretrained_teacher_dir_T3)

  ## Load the pretrained model and tokenizer
  print('---------Train the original teacher model!')
  esm_model = EsmModel.from_pretrained("facebook/esm2_t6_8M_UR50D")

  # Freeze all parameters in the ESM model
  for param in esm_model.parameters():
      param.requires_grad = False

  model = ProteinClassifier(esm_model, args.hidden_size, args.num_classes)
  

  model = model.to(device)
  
  print("-----Teacher model structure:")
  print(model)
  
  def count_model_parameters(model):
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total_params, trainable_params

  print("model parameters (total/trainable): ", count_model_parameters(model))

  # Define the loss function and optimizer
  #loss_function = nn.MSELoss()
  loss_function = nn.CrossEntropyLoss()
  optimizer = torch.optim.Adam(model.classifier.parameters(), lr=args.lr, weight_decay=1e-5)
  
  # Run the training loop
  random.seed(2024)
  train_MSE_list = []; val_MSE_list = []; test_MSE_list = []; 
  train_acc_list = []; val_acc_list = []; test_acc_list = []; 

  student_dir_save = args.student_dir
  os.makedirs(student_dir_save,exist_ok=True)
  os.makedirs(args.LMC_params,exist_ok=True)
  student_dir_save = Path(student_dir_save)
  
  # Early stopping parameters
  patience = 200  # Number of epochs to wait for improvement
  best_val_loss = float('inf')
  early_stop_counter = 0
  
  print('--------Training starts!')
  start_time=time.time()

  for epoch in range(0,args.epoch):
    if epoch<args.KD_epoch:
      train_MSE, train_acc = train_one_epoch_with_KD_mix(trainloader=train_loader, model_upd=model,
                                          loss_function=loss_fn_kd_mix_weighted, optimizer=optimizer, device=device, alpha=args.alpha, T=args.T)
      # model_MSE = train_one_epoch(train_loader, model, loss_function, optimizer, device)
      train_MSE_list.append(train_MSE)
      train_acc_list.append(train_acc)
    else: 
      train_MSE, train_acc, model_param = sample_par_with_KD_mix(trainloader=train_loader, model_upd=model, 
                                                      loss_function=loss_fn_kd_mix_weighted, optimizer=optimizer, device=device, epsilon_1=args.epsilon_1, epsilon_2=args.epsilon_2, alpha=args.alpha, T=args.T, ssize=NN)
      train_MSE_list.append(train_MSE)
      train_acc_list.append(train_acc)
      par_dir = args.LMC_params + "LMC_params_epoch"+str(epoch)+".pth"
      torch.save(model_param, par_dir)
    
    val_MSE, val_acc = test_one_epoch_with_KD_mix(val_loader, model, loss_fn_kd_mix_weighted, device=device, alpha=args.alpha, T=args.T)
    val_MSE_list.append(val_MSE)
    val_acc_list.append(val_acc)
    
    test_MSE, test_acc = test_one_epoch_with_KD_mix(test_loader, model, loss_fn_kd_mix_weighted, device=device, alpha=args.alpha, T=args.T)
    test_MSE_list.append(test_MSE)
    test_acc_list.append(test_acc)

    print('EPOCH:%d  Train Loss:%.4f  Train acc:%.4f  Val Loss acc:%.4f  Val acc:%.4f  Test Loss:%.4f   Test acc:%.4f   Whole Used Time:%s'
          %(epoch, train_MSE, train_acc, val_MSE, val_acc, test_MSE, test_acc, str(datetime.timedelta(seconds=int(time.time()-start_time)))))
    
    # Check for improvement in validation loss
    if val_MSE < best_val_loss:
      best_val_loss = val_MSE
      early_stop_counter = 0
      # Save the best model
      best_model_path = student_dir_save / 'best_model.pth'
      torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        }, best_model_path)
      print(f"Epoch {epoch}: New best model saved with Validation Loss: {best_val_loss:.4f}")
    else:
      early_stop_counter += 1
      # print(f"Epoch {epoch}: No improvement in Validation Loss for {early_stop_counter} epochs.")

    # Early stopping condition
    if early_stop_counter >= patience:
        print("Early stopping triggered. Training stopped.")
        break
    #---
    
    if (epoch+1)%20==0:
      checkpoint_path = student_dir_save / 'student_{}epoch.pth'.format(epoch+1)

      torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        }, checkpoint_path)
    
  #
  print('------Training ends!')
  
  ## plot
  plot_dir = args.path_plot 
  os.makedirs(plot_dir, exist_ok=True)

  fig, axes = plt.subplots(1, 2, figsize=(12, 5))

  # Accuracy subplot
  axes[0].plot(train_acc_list, color='black', label='train', linewidth=1, linestyle='-')
  axes[0].plot(val_acc_list, color='blue', label='val', linewidth=1, linestyle='-.')
  axes[0].plot(test_acc_list, color='red', label='test', linewidth=1, linestyle='--')
  axes[0].legend()
  axes[0].set_xlabel('Epoch')
  axes[0].set_ylabel('Accuracy')
  axes[0].set_title('Accuracy vs Epoch')

  # MSE subplot
  axes[1].plot(train_MSE_list, color='black', label='train', linewidth=1, linestyle='-')
  axes[1].plot(val_MSE_list, color='blue', label='val', linewidth=1, linestyle='-.')
  axes[1].plot(test_MSE_list, color='red', label='test', linewidth=1, linestyle='--')
  axes[1].legend()
  axes[1].set_xlabel('Epoch')
  axes[1].set_ylabel('MSE')
  axes[1].set_title('MSE vs Epoch')
  
  plt.tight_layout()
  plt.savefig(plot_dir+'student_LMC_mix_weighted_tr_v4')
  plt.show()
  plt.close()
  
  # Get end time
  current_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
  print("end time: ", current_time)
  print("++++++++++++++++++++++++++")
  
  
  
  




##---- functions --------------------------------------------------
# Define the dataset
class ProteinDataset_KD(Dataset):
    def __init__(self, dataframe, tokenizer, teacher_probs1=None, teacher_probs2=None):
        """
        Args:
            dataframe (DataFrame): Original data containing sequences and labels.
            tokenizer: Tokenizer for the sequences.
            teacher_probs1 (DataFrame): Class probabilities from teacher model 1.
            teacher_probs2 (DataFrame): Class probabilities from teacher model 2.
        """
        self.sequences = dataframe['Sequence'].values
        self.labels = dataframe['Label'].values
        self.tokenizer = tokenizer
        self.teacher_probs1 = teacher_probs1
        self.teacher_probs2 = teacher_probs2

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        sequence = self.sequences[idx]
        label = self.labels[idx]

        # Tokenize the input sequence
        inputs = self.tokenizer(
            sequence,
            return_tensors="pt",
            padding="max_length",  
            truncation=True,       
            max_length=1000        
        )
        # Fetch teacher probabilities if provided
        prob1 = torch.tensor(self.teacher_probs1.iloc[idx].values, dtype=torch.float) if self.teacher_probs1 is not None else None
        prob2 = torch.tensor(self.teacher_probs2.iloc[idx].values, dtype=torch.float) if self.teacher_probs2 is not None else None

        return inputs, torch.tensor(label, dtype=torch.long), prob1, prob2



def loss_fn_kd_mix(outputs, labels, outputs_T1, outputs_T2, alpha=0.5, T=8, device='cuda'):
  """
  Compute the knowledge-distillation (KD) loss given outputs, labels.
  "Hyperparameters": temperature and alpha
  NOTE: the KL Divergence for PyTorch comparing the softmaxs of teacher
  and student expects the input tensor to be log probabilities! See Issue #2
  """
  
  # labels = labels[:,0].type(torch.LongTensor).to(device)
  labels = labels.to(device)
  KD_loss = nn.KLDivLoss(reduction="batchmean")(F.log_softmax(outputs/T, dim=1),
                            F.softmax(outputs_T1/T, dim=1)) * (alpha) + \
            nn.KLDivLoss(reduction="batchmean")(F.log_softmax(outputs/T, dim=1),
                            F.softmax(outputs_T2/T, dim=1)) * (alpha) + \
            2*F.cross_entropy(outputs, labels) * (1. - alpha)

  return KD_loss




def entr(prob):
  tmp = torch.exp(prob - torch.max(prob, dim=1, keepdim=True)[0])
  prob = tmp / tmp.sum(dim=1, keepdim=True)
  entropy = -torch.sum(prob * torch.log2(prob + 1e-9), dim=1)  # Sum across columns
  return entropy


from torch.special import gammaln  # PyTorch has gammaln, which is log(gamma)
def multivariate_beta_torch(alpha):
    log_numerator = torch.sum(gammaln(alpha), dim=1)
    log_denominator = gammaln(torch.sum(alpha, dim=1))
    
    return torch.exp(log_numerator - log_denominator)

def multivariate_beta_log(alpha):
    return torch.sum(gammaln(alpha), dim=1) - gammaln(torch.sum(alpha, dim=1))


def loss_fn_kd_mix_weighted(outputs, labels, outputs_T1, outputs_T2, alpha=0.5, T=8, device='cuda'):
  """
  Compute the loss 
  "Hyperparameters": temperature and alpha
  """
  
  labels = labels.to(device)
  
  tmp1 = entr(outputs_T1); tmp2 = entr(outputs_T2); 
  aa = tmp1 + tmp2 + 1e-9
  w1 = (1 - tmp1 / aa).detach()
  w2 = (1 - tmp2 / aa).detach()
  
  weights = torch.stack([w1, w2], dim=0)  # Shape [2, 32]
  normalized_weights = weights / weights.sum(dim=0, keepdim=True)
  
  lambda_const = 5
  
  CE_loss_1 = ( lambda_const*F.softmax(outputs_T1 / T, dim=1) * F.log_softmax(outputs / T, dim=1)).sum(dim=1)
  CE_loss_2 = ( lambda_const*F.softmax(outputs_T2 / T, dim=1) * F.log_softmax(outputs / T, dim=1)).sum(dim=1)
  
  ##---
  Beta_1_log = multivariate_beta_log(1 + lambda_const * F.softmax(outputs_T1/T, dim=1) )
  Beta_2_log = multivariate_beta_log(1 + lambda_const * F.softmax(outputs_T2/T, dim=1) )
  l1 = torch.exp(-Beta_1_log.to(device) + CE_loss_1)
  l2 = torch.exp(-Beta_2_log.to(device) + CE_loss_2)
  
  ll_weighted = normalized_weights[0, :] * l1 + normalized_weights[1, :] * l2 
  
  CE_loss_weighted = -0.002*torch.log(ll_weighted) + 1
  ##---
  
  KD_loss = CE_loss_weighted * alpha + F.cross_entropy(outputs, labels, reduction="none")
  
  return KD_loss.mean()


def sample_par_with_KD_mix(trainloader, model_upd, loss_function, optimizer, device, epsilon_1=1e-1, epsilon_2=1e-3, alpha=0.5, T=2, ssize=100):
  model_upd.train()
  
  #pars_old = list(model_upd.parameters())
  #print("---------pars_old:", pars_old)
  
  # current_loss = 0; current_acc = 0
  total_loss = 0
  correct_predictions = 0
  total_samples = 0
  pars_new_list = []; 
  
  original_state = copy.deepcopy(model_upd.state_dict())
  
  # Iterate over the DataLoader for training data
  for batch in trainloader:
    # Get and prepare inputs
    inputs, labels, output_T1, output_T2 = batch
    # Move data to the correct device
    inputs = {key: value.squeeze(0).to(device) for key, value in inputs.items()}
    labels, output_T1, output_T2 = labels.to(device), output_T1.to(device), output_T2.to(device)
    
    model_upd.load_state_dict(copy.deepcopy(original_state))
    # Zero the gradients
    optimizer.zero_grad()
    # Perform forward pass
    output_1 = model_upd(inputs)
    
    loss = loss_function(outputs=output_1, labels=labels, outputs_T1=output_T1, outputs_T2=output_T2, alpha=alpha, T=T, device=device)
    
    # Perform backward pass
    loss.backward()
    # Perform optimization
    # optimizer.step()
    
    # Print statistics
    total_loss += loss.item()
    preds = torch.argmax(output_1, dim=1)  # Get the predicted class
    correct_predictions += (preds == labels).sum().item()  # Count correct predictions
    total_samples += labels.size(0)  # Update total sample count
    # print("loss:", loss)
  
    #print('-----------get grad:')
    
    ## ---start update the parameters:
    with torch.no_grad():
      for param in model_upd.classifier.parameters():
        # print("param.shape:", param.shape)
        noise = torch.normal(0,1,size=param.shape).to(device)
        tmp = -epsilon_1*ssize*param.grad + epsilon_2*noise
        param.add_(tmp)

    original_state = copy.deepcopy(model_upd.state_dict())
    #---------------
    pars_new = copy.deepcopy(model_upd.state_dict())
    #print("---------pars_new:", pars_new)
    pars_new_list.append(pars_new)
  
  # Calculate average loss and accuracy for the epoch
  avg_loss = total_loss / len(trainloader)
  accuracy = correct_predictions / total_samples
  return avg_loss, accuracy, pars_new_list





##--------------------------------------
if __name__ == '__main__':
  parser = argparse.ArgumentParser('BKD training and evaluation script', parents=[get_args_parser()])
  args = parser.parse_args()
  # if args.output_dir:
  #     Path(args.output_dir).mkdir(parents=True, exist_ok=True)
  main(args)


