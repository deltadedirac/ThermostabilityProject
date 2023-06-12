"""import re
import pandas as pd
import numpy as np
import torch, torch.nn
from transformers import EsmModel, EsmTokenizer
from optimum.bettertransformer import BetterTransformer
from tqdm.auto import tqdm
from .utilities import pooling_in_place
"""
import torch
from .baseEmbedding import baseEmbedding
from transformers import EsmModel, EsmTokenizer
from optimum.bettertransformer import BetterTransformer
import esm
from tqdm.auto import tqdm

class ESM2embeddings(baseEmbedding):
    
    def __init__(self, 
                 type_embedding = "facebook/esm2_t30_150M_UR50D",
                 device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu'),
                 type_tool = 'FacebookESM2',
                 max_memory_mapping = {0: "10GB", 'cpu': "8GB"} ) -> None:
    
        super(ESM2embeddings,self).__init__(  
                                            type_embedding = "facebook/esm2_t30_150M_UR50D",
                                            device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu'),
                                            max_memory_mapping = {0: "10GB", 'cpu': "8GB"}
                                            )
        self.device = device
        self.type_tool=type_tool
        
        if self.type_tool=='FacebookESM2' or self.type_tool!='huggingface':
            # Load ESM-2 model
            self.model, self.alphabet = esm.pretrained.esm2_t33_650M_UR50D() #esm2_t6_8M_UR50D()
            self.model = self.model.to(self.device)
            self.tokenizer = self.alphabet.get_batch_converter()
            
        else:
            self.tokenizer = EsmTokenizer.from_pretrained(type_embedding, do_lower_case=False)
            self.model =   EsmModel.from_pretrained(type_embedding, 
                                                        low_cpu_mem_usage=True, 
                                                        max_memory=max_memory_mapping#,
                                                        #output_hidden_states = True
                                                ).to(device=self.device)

        if torch.cuda.is_available() and self.device!='cpu':
            self.model = self.model.half()
            #self.model = self.model.eval()
    
    def truncate(self, sequences, length):
        '''Function to truncate protein sequences at a given length'''
        num_truncated = len([seq for seq in sequences if len(seq) > length])
        print(num_truncated, 'sequences were too long and have been truncated to', length, 'AA')
        sequences = [seq[:length] for seq in sequences]
        
        return sequences

    def esm2embedding(self, all_data, device, truncate_length=1000, layer_index=6, pt_batch_size=16, folder_path='../prepro_embeddings/esm2_embeddings'):
        
        embeddings = torch.tensor([]).to(device)
        all_data.sequence = self.truncate(all_data.sequence.values, truncate_length)
        
        for i in tqdm(range(0,len(all_data), pt_batch_size)):
            batch = all_data.iloc[i:i+pt_batch_size]

            esm_data = list(zip(batch.index, batch.sequence))
            batch_labels, batch_strs, batch_tokens = self.tokenizer(esm_data)
            batch_lens = (batch_tokens != self.alphabet.padding_idx).sum(1)

            batch_tokens = batch_tokens.to(device)

            with torch.no_grad():
                results = self.model(batch_tokens, repr_layers=[layer_index])
            token_embeddings = results["representations"][layer_index]

            # Generate per-sequence representations via averaging
            # NOTE: token 0 is always a beginning-of-sequence token, so the first residue is token 1.
            for j, tokens_len in enumerate(batch_lens):
                torch.save(token_embeddings[j].cpu().detach(), f"{folder_path}/{str(i+j)+'_'+str(tokens_len.item())+'_'+ batch.iloc[j].protein_id }.pt")
                sequence_embeddings = token_embeddings[j, 1 : tokens_len - 1].mean(0).reshape(1,-1)
                embeddings = torch.cat([embeddings, sequence_embeddings])
            
            torch.cuda.empty_cache()
                    
        return embeddings


