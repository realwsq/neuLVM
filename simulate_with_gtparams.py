import numpy as np
import copy
import matplotlib.pyplot as plt

from src.activity import simulate_and_plot_activity
from src.helper import get_best_trained_files
import pdb, glob
import pickle, os

model_params = dict(
    dt=0.2 * 1e-3,
    dt_meso = 0.004,
    time_end=200.,
    N=[400,400,200], 
    SAMPLED_NEURON_NUM=30,
    M=3, 

    # LIF neuron related parameters
    resting_potential=[36./2.5]*3,
    firing_thres=[15./2.5-np.log(10)]*3, 
    J=[9.984, 9.984, 19.968], 
    conmat=[[1, 0., 1], [0., 1, 1], [-1, -1, -1]], 
    A0=[5,20,25],
    alpha_mem=[50]*3, 
    alpha_syn=[1000/3., 1000/3., 500/3.], 
    refractory_t=0.004, 
    eps=1.,
    lambda_t=[1]*3, 

    initialize='stationary',
    a_cutoff=1.0, 
)
SAVE_DIR = './dataset/'


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
    plot_micro=True,
    save_micro=True,
    plot_meso_lif =False,
    sim_lif = True,
    save_sim_lif = False, 
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

