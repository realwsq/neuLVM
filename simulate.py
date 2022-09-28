import numpy as np
import copy
import matplotlib.pyplot as plt

from activity import simulate_and_plot_activity
import pdb, glob
import pickle, os

load_from_trained_model = True

if load_from_trained_model:
    et = 65
    '''
        select the trained file with the lowest loss
    '''
    best_loss = 1e9
    best_hist = best_best = None
    for hist in ['train1', 'train2', 'train3', 'train4', 'train5']:
        _subfolder = f'3pop example/data{et}/{hist}'
        _SAVE_DIR = f'./result/{_subfolder}'
        print(_SAVE_DIR)

        _pb = glob.glob(os.path.join(_SAVE_DIR, 'minimizor_result_[0-9]'))
        _pb = [int(_f.split('_')[-1]) for _f in _pb]
        best = np.max(_pb)
        est_model = pickle.load(open(f'{_SAVE_DIR}/minimizor_result_{best}', 'rb') )
        print(et, hist, best, est_model['log_loss'][-1])
        if est_model['log_loss'][-1] < best_loss:
            best_hist = hist
            best_best = best
            best_loss = est_model['log_loss'][-1]
    print("============== best ==============")
    print(et, best_hist, best_best, best_loss)
    subfolder = f'3pop example/data{et}/{best_hist}'
    SAVE_DIR = f'./result/{subfolder}'
    print(SAVE_DIR)

    '''
        read out trained parameters
    '''
    est_model = pickle.load(open(f'{SAVE_DIR}/minimizor_result_{best_best}', 'rb') )
    x = est_model['log_X'][-1]
    print(x, est_model['log_loss'][-1])
    numJ = 3 if len(x) == 12 else 9

    model_params = dict(
        dt=0.001,
        dt_meso = 0.004,
        time_end=100.,
        N=[400,400,200],
        SAMPLED_NEURON_NUM=3,
        M=3, 

        # LIF neuron related parameters
        alpha_mem = x[0:3], 
        J = x[3:3+numJ],
        resting_potential = x[3+numJ:6+numJ], 
        firing_thres= x[6+numJ:9+numJ], 
        conmat=[[1, 0., 1], [0., 1, 1], [-1, -1, -1]], 
        A0=[5,20,25],
        alpha_syn=[1000/3., 1000/3., 500/3.],
        refractory_t=0.004, 
        eps=1.,
        lambda_t=[1.]*3,
        syn_delay=0,
        initialize='stationary',
        a_cutoff=1.0, 
    )
else: 
    model_params = dict(
        dt=0.2 * 1e-3,
        dt_meso = 0.004,
        time_end=100.,
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
        plot_micro=not load_from_trained_model,
        save_micro=False,
        plot_meso_lif =load_from_trained_model,
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

