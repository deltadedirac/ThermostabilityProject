
import pandas as pd
import numpy as np
import torch, os
from tqdm.auto import tqdm
import matplotlib.pyplot as plt
import os.path
from os import path


def train_test_validation_splits(df):
    test,train_tot = df.loc[df['set']=='test'],df.loc[df['set']=='train']
    train, val = train_tot.loc[train_tot['validation']!=True], train_tot.loc[train_tot['validation']==True]
    return train,val, test

''' Method for creating batches from a list of objects, useful for making the training by batches.
    it was necessary to do so, because the BERT embeddings just give as outputs a list of elements'''
def build_batch_iterator_sequences(sequences_total, batch_size):

    long_list = sequences_total
    sub_list_length = batch_size
    sub_lists = [
        long_list[i : i + sub_list_length]
        for i in range(0, len(long_list), sub_list_length)
    ]
    return sub_lists

def prepare_train_test_val_seqs_by_batches(train, test, val, batch_size=16):
    batch_train = build_batch_iterator_sequences(train, batch_size)
    batch_test = build_batch_iterator_sequences(test, batch_size)
    batch_val = build_batch_iterator_sequences(val, batch_size)
    return batch_train, batch_test, batch_val


def pooling_in_place(batchtensor):
  batchtensor = batchtensor.permute(0,2,1)
  GlobalAvgpooling = torch.nn.AvgPool1d(kernel_size = batchtensor.size(2) , stride = 1, padding = 0)
  pooled_seq = GlobalAvgpooling(batchtensor)
  return pooled_seq


def pooling_and_final_representation(path):
  # Due to npz compression on each embedded sequence
  sample = np.load(path)
  embedded_seq = torch.from_numpy( sample[sample.files[0]] ).to(torch.float32)
  pooled_seq = pooling_in_place(embedded_seq)

  return pooled_seq 

#the test
def pooled_set_of_sequences(folder_path):
  #import pdb; pdb.set_trace()
  list_embeddings = []
  iter_paths = os.listdir(folder_path)
  for path in tqdm(iter_paths):
    list_embeddings.append( pooling_and_final_representation(folder_path + path) )

  #import pdb;pdb.set_trace()
  representation = torch.stack(list_embeddings)
  representation = representation.view(representation.size(0),-1)
  return representation


def tensor2dataloader(tensor_data, tensor_target, batch_size=50) -> torch.utils.data.DataLoader :
    Dataset = torch.utils.data.TensorDataset(tensor_data, tensor_target )
    Data_Loader = torch.utils.data.DataLoader(Dataset, batch_size=batch_size, shuffle=True)
    return Data_Loader


def plot_results( outcome, test_labels):

    indexes = list(range(0,len(test_labels) ) ) 

    plt.figure(figsize=(25,8))
    plt.plot(indexes, outcome.flatten().detach().cpu().tolist(), c='red', label="predicted")
    plt.plot( indexes, test_labels.detach().cpu().tolist(), c='green', label="ground truth")
    plt.legend(loc="upper left")
    #plt.ylim(-1.5, 2.0)
    plt.show()
    
def create_folder_embeddings_savings(tmpfiles):
    train_tmp_folder = tmpfiles+'train_embeddinds/'
    test_tmp_folder = tmpfiles+'test_embeddinds/'
    val_tmp_folder = tmpfiles+'val_embeddinds/'

    import os.path
    from os import path

    if path.exists(tmpfiles) == False:
        os.mkdir(tmpfiles)
        os.mkdir(train_tmp_folder )
        os.mkdir(test_tmp_folder )
        os.mkdir(val_tmp_folder )
    else:
        if path.exists(train_tmp_folder) == False or path.exists(test_tmp_folder) == False or path.exists(val_tmp_folder) == False:
            try:
                os.mkdir(train_tmp_folder )
                os.mkdir(test_tmp_folder )
                os.mkdir(val_tmp_folder )
            except Exception:
                pass