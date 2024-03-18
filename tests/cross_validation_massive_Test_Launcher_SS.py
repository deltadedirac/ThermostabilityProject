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
            "--time=5-00:00:00",
            "papermill", "{}".format(py_file), "{}".format(out_file),
            "-p", "config_path", "{}".format(config_path),
            "--log-output", "--log-level", "DEBUG", "--progress-bar" 
        ] )

def run_organisms_experiments(folder_path_out,
                              job_name,
                              config_file,
                              py_file,
                              list_bacteria):
    
    if not os.path.exists(folder_path_out):
        os.makedirs(folder_path_out)
        print("Directory '%s' created" %folder_path_out)
        
    '''
        - loss1: just the first chunck of loss of bias example, sumation of 2
                outputs with target in MSE
    '''
    for i in list_bacteria:
        launch_process(job_name=job_name+'_'+i, 
                       py_file=py_file, 
                       out_file=folder_path_out+i+'.ipynb', 
                       config_path=config_file+i+'.yaml', 
                       nodelist='pika', 
                       #gpures='gpu:L40:1',
                       gpures='shard:40',  
                       cpu_res='25')
        print(f'Running {i} on slurm')
        
        
        
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
    
    
    
    '''----------------------------------------------------------------------------------------------'''
    '''----------------------------------------------------------------------------------------------'''    
    ''' 
        Species vs Species experiment using dual loss with global avgtm with/without cleaning
        and Just MSE as loss Function.
    
    '''
    '''----------------------------------------------------------------------------------------------'''
    '''----------------------------------------------------------------------------------------------'''
    py_file = "18_LAMLP_CrossVal_Massive_test_all_taxonomy_23_Jan_2024.ipynb"
    config_path_dualnoclean = 'configs/cross_validation/species_experiments_loss_composed_globalAvgtm_speedup/loss2/'
    
    run_organisms_experiments(folder_path_out="crossval_metrics/metrics_per_species_dualloss_noclean_speedup/ipynb/",
                              job_name = "sgal_SSdual_",
                              config_file = config_path_dualnoclean,
                              py_file = py_file,
                              list_bacteria = list_bacteria)
    
    config_path_dualclean = 'configs/cross_validation/species_experiments_loss_composed_globalAvgtm_cleanup/loss2/'
    run_organisms_experiments(folder_path_out="crossval_metrics/metrics_per_species_dualloss_clean/ipynb/",
                              job_name = "sgal_SSdualclean_",
                              config_file = config_path_dualclean,
                              py_file = py_file,
                              list_bacteria = list_bacteria)
    
    config_path_MSE = 'configs/cross_validation/species_experiments_loss_MSE/MSE_alone/'

    
    run_organisms_experiments(folder_path_out="crossval_metrics/metrics_per_species/ipynb/",
                              job_name = "sgal_SS_MSE_",
                              config_file = config_path_MSE,
                              py_file = py_file,
                              list_bacteria = list_bacteria)
    
    '''----------------------------------------------------------------------------------------------'''
    '''----------------------------------------------------------------------------------------------'''
    '''----------------------------------------------------------------------------------------------'''
