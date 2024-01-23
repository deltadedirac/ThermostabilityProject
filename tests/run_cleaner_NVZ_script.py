import subprocess, ipdb, os
import yaml
import pandas as pd

def launch_process_cleaning(job_name, py_file, path_file, n_clusters, seq_col,nodelist='pika', 
                   gpures='gpu:L40:1', cpu_res='15'):
    # For some reason, I have to modify slightly the script because sbatch says that 
    # in the normal basis the script is not recognisible, so I have to run it using 
    # wrap to create synthetically sh inline, so that, it can run
    import ipdb; ipdb.set_trace()
    script_inline =  "python {} {} --scatter-plot --sequence_col {} ".format(py_file, path_file, seq_col) \
                                                + "--max_clusters {}".format(n_clusters)
    
    sbatch_config = \
        'sbatch --partition=high --nodelist={} --gres={} --job-name={} '.format(nodelist, gpures, job_name)\
            + '--ntasks=1 --cpus-per-task={} --time=3-00:00:00 --wrap="{}"'.format(str(cpu_res), script_inline)
    
    # Due to the nature of NVZ script, I have to add cd path just 
    # to make sure the results are gonna be located in the folder per 
    # species or any other output folder you specify.
    os.system('cd {} && '.format('/'.join(path_file.split('/')[:-1]))
               + sbatch_config)       

    

def split_df_to_file_per_species(df, species, path_out):
    df_species = df[df.key==species]
    print('saving {} in a file to be processed'.format(species))
    '''
        In case you want the fasta file per species
    
    with open(path_out.split('.tsv')[0]+'.fasta','a') as fasta_writer:
        for k, v in enumerate(df_species.sequence):
            fasta_writer.write(f'>{k}\n{v}\n')
    '''
    df_species.to_csv(path_out, sep='\t', index=False, header=True)
    return df_species
    
def exist_cluster_file(folder_cluster, tag='temp.clusters.'):
    flag=False
    match=[]
    for fname in os.listdir(folder_cluster):
        if 'temp.clusters.' in fname:
            flag=True
            match.append(fname)
            
            break
        
    return flag, match
    
if __name__ == '__main__':

    path_treated_meltome_filtered_key = '../datasets/global_cleanup_meltome_all_set/'
    path_meltome_file_with_key='train_MeltomeMix_for_cleaning_with_key_column.tsv'
    folder_path_out = '../datasets/clean_up_NVZ_per_species_MeltomeMix/'
    # the code from NVZ only accepts absolute paths for being executed 
    absolute_root_path = '/z/home/sgal/ML_Projects/ThermostabilityProject/'

    #ipdb.set_trace()
    
    df_pre_filtered = pd.read_csv(path_treated_meltome_filtered_key
                                  +
                                  path_meltome_file_with_key, sep="\t")
    
    list_species = df_pre_filtered.key.unique()

    import ipdb; ipdb.set_trace()
    list_dfs=[]
    for i in list_species:
           
        if not os.path.exists(folder_path_out + i):
            os.makedirs(folder_path_out + i)
            print("Directory '%s' created" %(folder_path_out+i))
            split_df_to_file_per_species(df_pre_filtered, i, folder_path_out+i+'/'+i+'.tsv')
            
        if os.path.exists(folder_path_out+i+'/'+i+'.tsv'):
            flag, index = exist_cluster_file(absolute_root_path +
                                                folder_path_out.split('../')[1]+i+'/' )
            if not flag:
                
                print('launching cleaning for {}'.format(folder_path_out+i+'/'+i+'.tsv'))
                
                launch_process_cleaning(job_name='sgal_clr_{}'.format(i), 
                                        py_file='/z/home/jrsx/git/deepfamily/src/clustering_pipeline.py',
                                        path_file=absolute_root_path + folder_path_out.split('../')[1] + i + '/' + i +'.tsv',
                                        n_clusters=30,
                                        seq_col='sequence',
                                        nodelist='pika', 
                                        gpures='gpu:1', 
                                        cpu_res='10')
                
                ''' Make cleanings per set using the cluster info'''
            else:
                df_clusters= pd.read_csv(folder_path_out+i+'/'+index[0], sep='\t')
                df_filter_species = pd.read_csv(folder_path_out+i+'/'+i+'.tsv', sep='\t')
                df_filter_species['clusters'] = df_clusters.cluster.tolist()
                df_filter_species = df_filter_species[df_filter_species.clusters!=-1]
                df_filter_species.to_csv( folder_path_out+i+'/'+i+'_clean.tsv', 
                                         sep='\t', index=False, header=True)
                list_dfs.append( df_filter_species.loc[:, df_filter_species.columns != 'clusters'] )
    
    final_df = pd.concat(list_dfs).sample(frac = 1)
    final_df.to_csv( folder_path_out+'AllMeltomeMix_clean_per_species.tsv', 
                                         sep='\t', index=False, header=True)
    print('done')
                
                
                

        
