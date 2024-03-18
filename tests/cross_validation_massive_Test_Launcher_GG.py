import subprocess, ipdb, os
import yaml


def launch_process(job_name, py_file, out_file, config_path, nodelist='pika', 
                   gpures='gpu:L40:1', cpu_res='35'):
    r = subprocess.run(
        [
            "sbatch", 
            "--partition=high", 
            "--nodelist={}".format(nodelist), 
            "--gres={}".format(gpures), 
            "--job-name={}".format(job_name),
            "--ntasks=1", 
            "--cpus-per-task={}".format(str(cpu_res)), 
            "papermill", "{}".format(py_file), "{}".format(out_file),
            "-p", "config_path", "{}".format(config_path),
            "--log-output", "--log-level", "DEBUG", "--progress-bar" 
        ] )


if __name__ == '__main__':
    
    ''' 
        Just Execution of experiments related to global prformance over FLIP-Meltome 
        dataset, i.e. Using all training set, as well as all validation/test set
    '''
    
    '''--------------------------------Global Global using dual loss without cleaning -----------------------------------------'''
    folder_path_GG_Dual_noclean = "crossval_metrics/metrics_global_loss_composed/Global/no_clean_speed/ipynb/"
    if not os.path.exists(folder_path_GG_Dual_noclean):
        os.makedirs(folder_path_GG_Dual_noclean)
        print("Directory '%s' created" %folder_path_GG_Dual_noclean)
        
    launch_process(job_name = "sgal_ggdual_noclean", 
                   py_file = "18_LAMLP_CrossVal_Massive_test_all_taxonomy_23_Jan_2024.ipynb", 
                   out_file = folder_path_GG_Dual_noclean + 'res_gg_noclean_dual.ipynb',
                   config_path = "configs/cross_validation/global_loss_composed_globalAvgtm/global_allmeltome_speedup.yaml", 
                   nodelist='ai', gpures='shard:40', cpu_res='25')
    '''-------------------------------------------------------------------------------------------------------------------------'''



    '''--------------------------------Global Global using dual loss with cleaning -----------------------------------------'''
    folder_path_GG_Dual_clean = "crossval_metrics/metrics_global_loss_composed/Global/no_clean_speed/ipynb/"
    if not os.path.exists(folder_path_GG_Dual_clean):
        os.makedirs(folder_path_GG_Dual_clean)
        print("Directory '%s' created" %folder_path_GG_Dual_clean)
        
    launch_process(job_name = "sgal_ggdual_clean", 
                   py_file = "18_LAMLP_CrossVal_Massive_test_all_taxonomy_23_Jan_2024.ipynb", 
                   out_file = folder_path_GG_Dual_clean + 'res_gg_clean_dual.ipynb',
                   config_path = "configs/cross_validation/global_loss_composed_globalAvgtm/global_allmeltome_cleaned_speedup.yaml", 
                   nodelist='ai', gpures='shard:40', cpu_res='25')
    '''-------------------------------------------------------------------------------------------------------------------------'''






    '''--------------------------------Global Global using single loss without cleaning -----------------------------------------'''
    
    folder_path_GG_MSE_noclean = "crossval_metrics/metrics_global_loss_MSE/Global/no_clean_speed/ipynb/"
    if not os.path.exists(folder_path_GG_MSE_noclean):
        os.makedirs(folder_path_GG_MSE_noclean)
        print("Directory '%s' created" %folder_path_GG_MSE_noclean)
        
    launch_process(job_name = "sgal_ggMSE_noclean", 
                   py_file = "18_LAMLP_CrossVal_Massive_test_all_taxonomy_23_Jan_2024.ipynb", 
                   out_file = folder_path_GG_MSE_noclean + 'res_gg_noclean_MSE.ipynb',
                   config_path = "configs/cross_validation/global_loss_MSE/global_allmeltome.yaml", 
                   nodelist='ai', gpures='shard:40', cpu_res='25')
    '''-------------------------------------------------------------------------------------------------------------------------'''



    '''--------------------------------Global Global using single loss with cleaning -----------------------------------------'''
    folder_path_GG_MSE_clean = "crossval_metrics/metrics_global_loss_MSE/Global/clean_speed/ipynb/"
    if not os.path.exists(folder_path_GG_MSE_clean):
        os.makedirs(folder_path_GG_MSE_clean)
        print("Directory '%s' created" %folder_path_GG_MSE_clean)
        
    launch_process(job_name = "sgal_ggMSE_clean", 
                   py_file = "18_LAMLP_CrossVal_Massive_test_all_taxonomy_23_Jan_2024.ipynb", 
                   out_file = folder_path_GG_MSE_clean + 'res_gg_clean_MSE.ipynb',
                   config_path = "configs/cross_validation/global_loss_MSE/global_allmeltome_cleaned.yaml", 
                   nodelist='ai', gpures='shard:40', cpu_res='25')
    '''-------------------------------------------------------------------------------------------------------------------------'''