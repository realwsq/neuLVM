import numpy as np
import pickle

from src.activity import simulate_and_plot_activity
from src.helper import get_best_trained_files, variablenamed_withorder

end_timepoint = 11
'''
    select the trained file with the lowest loss
'''
best_Mfile, best_Efile, SAVE_DIR = get_best_trained_files(f'./result/3pop example/data{end_timepoint}', 
                                            [f'train{i}' for i in range(5)])

'''
    read out trained parameters
'''
est_model = pickle.load(open(best_Mfile, 'rb') )
x = est_model['X']
print(f"--- Learned parameters ({variablenamed_withorder(3)})---:")
print(x)
print(f"--- Loss ---:")
print(est_model['loss_last'])

model_params = dict(
    dt=0.001,
    dt_meso = 0.004,
    time_end=100.,
    N=[400,400,200],
    SAMPLED_NEURON_NUM=3,
    M=3, 

    # LIF neuron related parameters
    alpha_mem = x[0:3], 
    J = x[3:6],
    resting_potential = x[6:9], 
    firing_thres= x[9:12], 
    conmat=[[1, 0., 1], [0., 1, 1], [-1, -1, -1]], 
    A0=[5,20,25],
    alpha_syn=[1000/3., 1000/3., 500/3.],
    refractory_t=0.004, 
    eps=1.,
    syn_delay=0,
    initialize='stationary',
    a_cutoff=1.0, 
)

input_params = dict(
    base_I=0.0, 
    I_ext=20, 
    I_ext_start=np.arange(0, model_params['time_end']-1.5, 1.0)+1.,
    I_last_time=0.5,
    I_type='none', #'random', 'Pozzorini'
    I_ylim=10, 
)
plot_params = dict(
    Nsim = 1,
    sim_meso = True,
    save_sim_meso = False, 
    w=80,

    savepath=SAVE_DIR,
    saveplot=True,
    usetex=False,
    noshow=True,
    font_size=21,
    
    ylim=(0, 45),
)

# Three plots with A_inf plot
simulate_and_plot_activity(
    model_params,
    input_params,
    **plot_params,
)

