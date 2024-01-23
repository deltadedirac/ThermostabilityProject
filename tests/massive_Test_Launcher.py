import subprocess, ipdb, os
import yaml


def launch_process(job_name, py_file, out_file, config_path, nodelist='pika', 
                   gpures='gpu:L40:1', cpu_res='15'):
    r = subprocess.run(
        [
            "sbatch", 
            "--partition=high", 
            "--nodelist={}".format(nodelist), 
            "--gres={}".format(gpures), 
            "--job-name={}".format(job_name),
            "--ntasks=1", 
            "--cpus-per-task={}".format(str(cpu_res)), 
            "--time=3-00:00:00",
            "papermill", "{}".format(py_file), "{}".format(out_file),
            "-p", "config_path", "{}".format(config_path),
            "--log-output", "--log-level", "DEBUG", "--progress-bar" 
        ] )


if __name__ == '__main__':

    list_bacteria = [ 'Caenorhabditis_elegans_lysate', 
                'Mus_musculus_BMDC_lysate',  
                'Danio_rerio_Zenodo_lysate', 
                'Geobacillus_stearothermophilus_NCA26_lysate', 
                'Mus_musculus_liver_lysate', 
                'Drosophila_melanogaster_SII_lysate', 
                'Arabidopsis_thaliana_seedling_lysate', 
                'Bacillus_subtilis_168_lysate_R1',  
                'Escherichia_coli_cells', 
                'Escherichia_coli_lysate', 
                'Oleispira_antarctica_RB-8_lysate_R1', 
                'Saccharomyces_cerevisiae_lysate', 
                'Thermus_thermophilus_HB27_cells', 
                'Thermus_thermophilus_HB27_lysate', 
                'Picrophilus_torridus_DSM9790_lysate', 
                'HepG2', 'HAOEC', 
                'HEK293T', 'HL60', 
                'HaCaT', 'Jurkat', 
                'pTcells', 
                'colon_cancer_spheroids',
                'U937', 'K562']

    #config_path = 'configs/global_vs_species_loss_composed/'
    #folder_path_out = 'metrics_global_vs_species_loss_composed/ipynbs/'
    
    
    '''----------------------------------------------------------------------------------------------'''
    '''----------------------------------------------------------------------------------------------'''    
    ''' 
        Global vs Species experiment using dual loss with global 
    
        Avgtm, both with and without cleaned train set                          
    '''
    '''----------------------------------------------------------------------------------------------'''
    '''----------------------------------------------------------------------------------------------'''
    ''' THE CORRECTION IS IN SPEEDUP STUFF, IGNORE THE OTHERS, JUST BEAR THEM IN MIND AS REFERENCES!!!'''
    #config_path = 'configs/global_vs_species_loss_composed_globalAvgtm_speedup/'
    #folder_path_out = 'metrics_global_vs_species_loss_composed_speedup/ipynbs/'


    #config_path = 'configs/species_experiments_loss_composed_globalAvgtm_cleanup/loss2/'
    #folder_path_out = 'metrics_species_vs_species_loss_composed_clean/ipynbs/'
    #config_path = 'configs/global_vs_species_loss_composed_globalAvgtm_cleanup/'
    #folder_path_out = 'metrics_global_vs_species_loss_composed_cleanup/'

    #py_file = '18_LAMLP_Massive_test_all_taxonomy_15_Oct_2023.ipynb'
    #out_file_lexem = 'res_18ipynb_global_vs_species_loss_two_outputs_'
    
    '''----------------------------------------------------------------------------------------------'''
    '''----------------------------------------------------------------------------------------------'''
    
    
    
    '''----------------------------------------------------------------------------------------------'''
    '''----------------------------------------------------------------------------------------------'''    
    ''' 
        Species vs Species experiment using dual loss with global 
    
        Avgtm, both with and without cleaned train set                          
    '''
    '''----------------------------------------------------------------------------------------------'''
    '''----------------------------------------------------------------------------------------------'''
    config_path = 'configs/species_experiments_loss_composed_globalAvgtm_speedup/loss2/'
    folder_path_out = 'metrics_per_species_dualloss_noclean_speedup/ipynbs/'

    #config_path = 'configs/species_experiments_loss_composed_globalAvgtm_cleanup/loss2/'
    #folder_path_out = 'metrics_per_species_dualloss_clean/ipynbs/'
    
    #config_path = 'configs/species_experiments_loss_composed_globalAvgtm/loss2/'
    #folder_path_out = 'metrics_per_species_dualloss_noclean/ipynbs/'

    py_file = '18_LAMLP_Massive_test_all_taxonomy_15_Oct_2023.ipynb'
    out_file_lexem = 'res_18ipynb_species_vs_species_loss_two_outputs_'
    
    '''----------------------------------------------------------------------------------------------'''
    '''----------------------------------------------------------------------------------------------'''
    
    
    

    
    
    
    
    
    '''----------------------------------------------------------------------------------------------'''
    '''----------------------------------------------------------------------------------------------'''    
    ''' 
        Species vs Species experiment using single loss                   
    '''
    '''----------------------------------------------------------------------------------------------'''
    '''----------------------------------------------------------------------------------------------'''

    #config_path = 'configs/global_vs_species_loss_MSE/'
    #folder_path_out = 'metrics_global_vs_species_loss_MSE/ipynbs/'

    #py_file = '18_LAMLP_Massive_test_all_taxonomy_loss_1_component.ipynb'
    #out_file_lexem = 'res_18ipynb_loss_MSE_'
    
    '''----------------------------------------------------------------------------------------------'''
    '''----------------------------------------------------------------------------------------------'''
    
    
    #config_path = 'configs/global_vs_species_loss_MSE/'
    #folder_path_out = 'metrics_global_vs_species_loss_MSE/ipynbs/'
    
    
    # This is for one single MSE loss between predicted tm and the ground truth
    #py_file = '18_LAMLP_Massive_test_all_taxonomy_loss_1_component.ipynb'
    #out_file_lexem = 'res_18ipynb_loss_MSE_'
    
    
    
    # This is for the composed MSE loss with two outputs and the Avg tm per organism
    #py_file = '18_LAMLP_Massive_test_all_taxonomy_15_Oct_2023.ipynb'
    #out_file_lexem = 'res_18ipynb_species_vs_species_loss_two_outputs_'
    #out_file_lexem = 'res_18ipynb_global_vs_species_loss_two_outputs_'
    #ipdb.set_trace()
    
    
    if not os.path.exists(folder_path_out):
        os.makedirs(folder_path_out)
        print("Directory '%s' created" %folder_path_out)
        
    '''
        - loss1: just the first chunck of loss of bias example, sumation of 2
                outputs with target in MSE
    '''
    for i in list_bacteria:
        launch_process(job_name='sgal_species_'+i, 
                       py_file=py_file, 
                       out_file=folder_path_out+out_file_lexem+i+'.ipynb', 
                       config_path=config_path+i+'.yaml', 
                       #nodelist='koala', 
                       #gpures='gpu:A5000:1',
                       nodelist='pika', 
                       #gpures='gpu:L40:1',
                       gpures='shard:40',  
                       cpu_res='30')
        print(f'Running {i} on slurm')
