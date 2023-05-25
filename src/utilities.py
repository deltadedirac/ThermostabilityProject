
import pandas as pd
import numpy as np
import torch, os
from tqdm.auto import tqdm
import matplotlib.pyplot as plt
import os.path
from os import path
from Bio.SeqIO.FastaIO import SimpleFastaParser

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
            

def load_full_meltome_FLIP_db(complete_meltome_db):
    with open(complete_meltome_db) as fasta_file:  # Will close handle cleanly
        identifiers = []
        sequences = []
        OGT = []
        for title, sequence in SimpleFastaParser(fasta_file):
            identifiers.append(title.split(None, 1)[0])  # First word is ID
            sequences.append(sequence)
            OGT.append( title.split(None,1)[1].split('=')[1] )
            
    full_meltome_db = pd.DataFrame( list(zip(identifiers, sequences, OGT)), columns=['protein_id', 'sequence', 'target'])
    return full_meltome_db


def seek_UniprotID_association_Meltome_prots(df, df_db):
    """
    list_ids = []
    for i in range(0,len(df)):
        print(i)
        if i==785:
            print('from here')
        query_seq = df.sequence.iloc[i]
        temp = df.target.astype(str).iloc[i]
        protein_id_seek = df_db.query('sequence==@query_seq & target==@temp').protein_id.item()#.tolist()
        #protein_id_seek = full_meltome_db.query('sequence==@data.sequence').protein_id.item()
        list_ids.append( protein_id_seek  )

    df['UniprotID']= list_ids
    return df
    """
    intersection = pd.merge(df.astype(str), df_db.astype(str), on=['sequence','target'],how='inner')
    intersection.protein_id = intersection.protein_id.apply(lambda x: x.split('_')[0] )
    return intersection


def get_url(url, **kwargs):
    response = requests.get(url,**kwargs)
      
    if not response.ok:
        #print(response.text)
        response.raise_for_status()
        #sys.exit()
        
    return response

# convert Uniprot accessions to uniprot id or primaryaccession to consume
def get_equivalence_UniprotID(WEBSITE_API, ID):
    r = get_url(f"{WEBSITE_API}/uniprotkb/{ID}")
    output = r.json()['primaryAccession']
    return output

import profet
from profet import Fetcher
from profet import alphafold
from profet import pdb
import requests, sys, json

def download_UniprotID_Alphafold_Structures(df, tmpdir, label_dir, label_name):
        
    #ONLY_ALPHAFOLD = "F4HvG8"
    #ONLY_PDB = "7U6Q"
    import io
    from contextlib import redirect_stdout
    
    WEBSITE_API = 'https://rest.uniprot.org/'
    PROTEINS_API = 'https://www.ebi.ac.uk/proteins/api'
    
    fetcher = Fetcher()
    fetcher.set_directory(str(tmpdir))
    fetcher.set_default_db("alphafold")
    Uniprot_ID_list =df.protein_id.tolist()
    

    open(label_dir+label_name+'.txt', "a").write("Original_ID\tUniprot_ID\tStructure_Alphafold\tSource\n")
    
    for data in tqdm(Uniprot_ID_list):
        Uniprot_ID = get_equivalence_UniprotID(WEBSITE_API, data)
        f = io.StringIO()

        try:
            with redirect_stdout(f):
                filename, _ = fetcher.get_file(Uniprot_ID, filesave=True, db='alphafold')
                out = f.getvalue().split('\n')[-2].split(':')[1].strip()
                open(label_dir+label_name+'.txt', "a").write(data+'\t'+Uniprot_ID+'\t'+filename+'\t'+out+'\n')
        except:
            open(label_dir+label_name+'.txt', "a").write(data+'\t'+Uniprot_ID+'\t'+'No structure available\tNA'+'\n')

'''----------------------------------------------------------------------------------------------------------'''
'''----------------------------------------------------------------------------------------------------------'''
'''                               Methods from Gustav to agilize reproducibility of his results              '''
'''----------------------------------------------------------------------------------------------------------'''
'''----------------------------------------------------------------------------------------------------------'''



def handle_nans(df, target_columns, method='remove', ):
    '''Function to remove NAs. Either by removing the whole row or by taking the mean of the other values'''
    if method == 'remove':
        obs_before_drop = len(df)
        df = df.dropna(subset=target_columns)
        obs_after_drop = len(df)
        print(obs_before_drop - obs_after_drop, 'observations were removed which had one or more unknown Tm')
        print('Final number of proteins:', obs_after_drop)
    
    elif method == 'mean':
        num_na = df.isna().sum().sum()
        df = df.T.fillna(df.mean(axis=1)).T
        obs_before_drop = len(df)
        df = df.dropna()
        obs_after_drop = len(df)
        print('Filled', num_na, 'Nan cells with mean values (Check that there are only numerical observations in the pH columns)')
        print(obs_before_drop - obs_after_drop, 'observations were removed which could not be interpolated')
        
    elif method == 'keep':
        print('Keeping all nan value')
        
    return df


from Bio import SeqIO
def load_fasta_to_df(filename):
    with open(filename) as fasta_file:  # Will close handle cleanly
        identifiers = []
        seqs = []
        for seq_record in SeqIO.parse(fasta_file, 'fasta'):  # (generator)
            identifiers.append(seq_record.id)
            seqs.append(str(seq_record.seq))
    #Gathering Series into a pandas DataFrame and rename index as ID column
    df = pd.DataFrame(dict(key=identifiers, sequence=seqs)).set_index(['key'])
    
    return df

'''----------------------------------------------------------------------------------------------------------'''
'''----------------------------------------------------------------------------------------------------------'''