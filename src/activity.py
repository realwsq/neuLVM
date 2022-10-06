import time
import copy
import os
import pickle

from scipy.ndimage import gaussian_filter1d
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import gridspec
import matplotlib

import tensorflow as tf
from src.LIFmesoCell import LIFmesoCell
from src.LIFmicroPop import LIFmicroPop
from src.helper import moving_average
import pdb

try:
    # Disable all GPUS
    tf.config.set_visible_devices([], 'GPU')
    visible_devices = tf.config.get_visible_devices()
    for device in visible_devices:
        assert device.device_type != 'GPU'
except:
    # Invalid device or cannot modify virtual devices once initialized.
    # pass
    print('not able to disable GPU')


def get_identity(**kwargs):
    id = ''
    for k in kwargs.keys():
        id = id + k + '_' + str(kwargs[k]) + '_'
    return id


def simulate_and_plot_activity(
    model_params,
    input_params,
    Nsim=1,
    plot_micro=True,
    save_micro=False,
    plot_meso_lif=False,
    sim_lif=True,
    save_sim_lif=False, 
    w=3,
    savepath='',
    saveplot=False,
    usetex=False,
    noshow=False,
    font_size="12",
    ylim=None,
    figsize=(10, 8),
    loc="best",
    font_family="serif",
):
    if usetex:
        plt.rc("text", usetex=True)
        plt.rc("font", family=font_family, size=font_size)
        matplotlib.rcParams["text.latex.preamble"] = [r"\usepackage{amsmath}"]

    M = model_params['M']
    N = model_params['N']
    SAMPLED_NEURON_NUM = model_params['SAMPLED_NEURON_NUM']
    dt = model_params["dt"]
    dt_meso = model_params["dt_meso"]
    a_cutoff = model_params['a_cutoff']
    T_steps = int(model_params['time_end']/dt)
    I_start = input_params["I_ext_start"]
    I_time = input_params['I_last_time']
    base_I = input_params["base_I"]
    I_ext = input_params["I_ext"]


    if plot_micro:
        if input_params['I_type'] == 'step':
            I_ext_vec = base_I * np.ones((T_steps, M))
            for _I_start in I_start:
                I_ext_vec[int(_I_start/dt):int((_I_start+I_time)/dt), -1] = (base_I+I_ext)
                # I_ext_vec[int(_I_start/dt):int((_I_start+I_time)/dt), 0] = (base_I+I_ext)
        elif input_params['I_type'] == 'sin':
            I_ext_vec = np.sin(np.array(range(T_steps))/T_steps*2 * np.pi) * np.abs(I_ext-base_I) * 0.7 + (I_ext+base_I)/2
        elif input_params['I_type'] == 'random':
            I_ext_vec = base_I * np.ones((T_steps, M))
            for _I_start in I_start:
                I_ext_vec[int(_I_start/dt):int((_I_start+I_time)/dt), :] = np.repeat(base_I+np.random.normal(0., I_ext, int(I_time/dt)), M).reshape((-1,M))
        elif input_params['I_type'] == 'Pozzorini':
            f_ext = 0.2
            I0 = base_I # base_I=I0
            sigma0_ext = I_ext
            Delta_sigma_ext = I_ext
            tau_ext = 0.003
            I_ext_vec = I0 * np.ones((T_steps, M))  
            for _I_start in I_start:
                _convert = int(dt_meso/dt)
                for Ti in range(int(_I_start/dt)+1, int((_I_start+I_time)/dt), _convert):
                    _I_last = I_ext_vec[Ti-1, 0]
                    _sigma_ext = sigma0_ext*(1+Delta_sigma_ext*np.sin(2*np.pi*f_ext*Ti*dt))
                    I_ext_vec[Ti:Ti+_convert, -1] = _I_last + (I0-_I_last)/tau_ext*dt+np.sqrt(2*_sigma_ext**2*dt/tau_ext)*np.random.normal()
                    I_ext_vec[Ti:Ti+_convert, :] = _I_last + (I0-_I_last)/tau_ext*dt+np.sqrt(2*_sigma_ext**2*dt/tau_ext)*np.random.normal()
        elif input_params['I_type'] == 'none':
            I_ext_vec = np.zeros((T_steps, M))

        # Particle simulation
        A_t_pops = []
        for ti in range(Nsim):
            # stimulate
            t=time.time()
            id = get_identity(microM=M,
                                N=N, 
                                time_end=model_params['time_end'],
                                rp=model_params['resting_potential'],
                                ft=model_params['firing_thres'],
                                J=model_params['J'], 
                                dt=dt, 
                                I_type=input_params['I_type'],
                                I_ext=input_params['I_ext'],
                                simi=ti)


            # LIF neuron
            ts_P, A_t_pop, sampled_spikes, sampled_spikeprobs = \
                        LIFmicroPop(**model_params, I_ext_vec=I_ext_vec)
            # ts_P: (T), 
            # A_t_pop: (T, M)
            # sampled_spikes, sampled_spikeprobs: (M, T, N)
            # GLM neuron
            # ts_P, A_t_pop, H, _, sampled_spikes, sampled_spikeprobs = particle_population_nomemory(**params_p, I_ext_vec=I_ext_vec, N=N, SAMPLED_NEURON_NUM=100)
            print(f"Particle simulation done in {time.time()- t:.2f}s")
            
            if save_micro: 
                # open a file, where you ant to store the data
                print(id)
                file = open(os.path.join(savepath, id), 'wb')
                pickle.dump({'ts_P':ts_P, 
                            'A_t_pop':A_t_pop,
                            'sampled_spikes':sampled_spikes[:,:,:SAMPLED_NEURON_NUM], 
                            'sampled_spikeprobs': sampled_spikeprobs[:,:,:SAMPLED_NEURON_NUM], 
                            'I_ext': I_ext_vec,},
                            file)
                file.close()


            A_t_pops.append(A_t_pop)

        A_t_pops = np.array(A_t_pops)
        sampled_spikes = np.expand_dims(sampled_spikes, 0)

        Aest_from_sampled_spikes = moving_average(sampled_spikes.mean(2)[0], 50, kernel='gaussian')/dt



    if plot_meso_lif:
        
        a_grid_size = int(a_cutoff/dt_meso)
        time_steps = int(model_params["time_end"]/dt_meso)
        ts_meso = np.linspace(0., model_params['time_end'], time_steps+1)[:-1]

        temp_Z_t_history = np.tile([0.]*(a_grid_size-1)+[1.]+[0.]*time_steps, (1,M,1)).astype(np.float32)
        sampled_spike_history = np.zeros(((1,M, a_grid_size+time_steps, SAMPLED_NEURON_NUM)), dtype=np.float32)
        sampled_spike_history[:,:, a_grid_size-1] = 1.
        rnn = LIFmesoCell.load_model(model_params['alpha_mem'], model_params['J'], model_params['resting_potential'], model_params['firing_thres'], temp_Z_t_history, 
                                model_params['alpha_syn'], model_params['eps'], model_params['refractory_t'], model_params['conmat'], model_params['lambda_t'], model_params['syn_delay'],
                                model_params['dt_meso'], a_grid_size, model_params['M'], model_params['N'], model_params['SAMPLED_NEURON_NUM'], time_steps, 1,
                                sampled_spike_history,
                                model_params['initialize'],
                                True,
                                True,
                                True
                                )


        if input_params['I_type'] == 'step':
            I_ext_vec_meso = base_I * np.ones(( int(model_params['time_end']/dt_meso), M))
            for _I_start in I_start:
                # print(int(_I_start/dt_meso),int((_I_start+I_time)/dt_meso))
                I_ext_vec_meso[int(_I_start/dt_meso):int((_I_start+I_time)/dt_meso), 0] = (base_I+I_ext)
        elif input_params['I_type'] == 'sin':
            I_ext_vec_meso = np.sin(np.array(range(time_steps))/time_steps*2 * np.pi) * np.abs(I_ext-base_I) * 0.7 + (I_ext+base_I)/2
        elif input_params['I_type'] == 'random':
            I_ext_vec_meso = base_I * np.ones((int(model_params['time_end']/dt_meso), M))
            for _I_start in I_start:
                I_ext_vec_meso[int(_I_start/dt_meso):int((_I_start+I_time)/dt_meso), 0] = base_I+np.random.normal(0., I_ext, int(I_time/dt_meso))
        elif input_params['I_type'] == 'Pozzorini':
            f_ext = 0.2
            I0 = base_I # base_I=I0
            sigma0_ext = I_ext
            Delta_sigma_ext = I_ext
            tau_ext = 0.003
            I_ext_vec_meso = I0 * np.ones((int(model_params['time_end']/dt_meso), M))  
            for _I_start in I_start:
                _convert = int(dt_meso/dt)
                for Ti in range(int(_I_start/dt_meso)+1, int((_I_start+I_time)/dt_meso)):
                    _I_last = I_ext_vec_meso[Ti-1, 0]
                    _sigma_ext = sigma0_ext*(1+Delta_sigma_ext*np.sin(2*np.pi*f_ext*Ti*dt))
                    I_ext_vec_meso[Ti, :] = _I_last + (I0-_I_last)/tau_ext*dt+np.sqrt(2*_sigma_ext**2*dt/tau_ext)*np.random.normal()
        elif input_params['I_type'] == 'none':
            I_ext_vec_meso = np.zeros((int(model_params['time_end']/dt_meso), M))
        I_ext_vec_meso = (I_ext_vec_meso.T)[np.newaxis,:,:]


        EA_t_ASMAmesos = []
        A_t_ASMAmesos = []
        messs = []
        for ti in range(Nsim):
            t = time.time()
            id = get_identity(ASMAmesoM=M,
                            N=N, 
                            I_ext=I_ext,
                            I_last_time=I_time,
                            # rp=model_params['resting_potential'][0],
                            # ft=model_params['firing_thres'][0],
                            # J=model_params['J'][0], 
                            # initialize=model_params['initialize'],
                            simi=ti+25)

            if sim_lif:
                rnn.cell.reset()
                EA_t_ASMAmeso, Z_t_ASMAmeso, _, _, _, mess, lambda_t = rnn(LIFmesoCell.out_to_in(I_ext=I_ext_vec_meso)['I_ext'])
                # (b, T, M, 1)
                EA_t_ASMAmeso = LIFmesoCell.in_to_out(EA_est=EA_t_ASMAmeso)['EA_est'][0].T   # (T,M)
                A_t_ASMAmeso = LIFmesoCell.in_to_out(Z_est=Z_t_ASMAmeso)['Z_est'][0].T / dt_meso # (T,M)
                mess = LIFmesoCell.in_to_out(mess=mess)['mess'][0].T #(T,M)
                sampled_spikes_meso = LIFmesoCell.in_to_out(sampled_hist_gt=rnn.cell.sampled_hist_gt)['sampled_hist_gt'][0,:,a_grid_size:]
                print(f'mean lambda_t: {np.mean(lambda_t)}')
                print(f'var mess: {np.var(mess)}')


                if save_sim_lif:
                    file = open(os.path.join(savepath, id), 'wb') 
                    pickle.dump({'ts_P':ts_meso, 
                                    'A_t_ASMAmeso':A_t_ASMAmeso, # (T, M)
                                    'EA_t_ASMAmeso':EA_t_ASMAmeso, # (T, M)
                                    'sampled_spikes':sampled_spikes_meso, #(M,T,Nsampled)
                                    'mess': mess, #(T,M)
                                    'I_ext': I_ext_vec_meso, #(T)
                                }, 
                                file)
                    file.close()
            else:
                 results = pickle.load(open(os.path.join(savepath, id), 'rb'))
                 EA_t_ASMAmeso = results['EA_t_ASMAmeso']
                 A_t_ASMAmeso = results['A_t_ASMAmeso']
                 mess = results['mess']

            print(f'mean mess: {np.mean(mess)}')
            print(f'mean lambda_t: {np.mean(lambda_t)}')
            print(f"ASMAmeso simulation done in {time.time()- t:.2f}s")
            EA_t_ASMAmesos.append(EA_t_ASMAmeso)
            A_t_ASMAmesos.append(A_t_ASMAmeso)
            messs.append(mess)
        
        A_t_ASMAmesos = np.array(A_t_ASMAmesos)
        EA_t_ASMAmesos = np.array(EA_t_ASMAmesos)
        messs = np.array(messs)
        EEA_t_ASMAmeso = np.mean(EA_t_ASMAmesos,axis=0)
        varA_t_ASMAmeso = np.var(A_t_ASMAmesos,axis=0)


    '''
        START TO PLOT
    ''' 

    if plot_micro:

        num_plots = 1 # spike trains
        num_plots += 1 # Empirical population activity of micro model 
        num_plots += 1 # dominant interval distribution => I_ext
        height_ratios = [1, 1, 1]

        fig = plt.figure(figsize=figsize)
        gs = gridspec.GridSpec(num_plots, 1, height_ratios=height_ratios)
        gs.update(hspace=0.0)  # remove gaps between subplots 
        plots = dict()

        # ''' 
        #     for spike trains
        # '''
        ax0 = plt.subplot(gs[0]) 

        # sampled_spikes = sampled_spikes[0] # use (last) one trial to plot
        # Nneuron = sampled_spikes.shape[2] # (M, T, N)
        # colors = 'black'
        # for m in range(M):
        #     for ni in range(Nneuron):
        #         ax0.eventplot(
        #                 ts_P[sampled_spikes[m,:,ni].astype(np.bool)],
        #                 lineoffsets=m*Nneuron+ni+1,
        #                 colors=colors,
        #                 # linewidths=0.5,
        #             )
        # ax0.set_ylim((0.5, M*Nneuron+0.5))
        # ax0.set_ylabel('neuron Id')


        '''
            Empirical population activity of micro model 
        '''
        ax1 = plt.subplot(gs[1]) #, sharex = ax0)
        # use (last) one trial to plot
        # A_t_pop
        A_t_pop = A_t_pops[-1] # (T, M)
        begin_idx = 0
        w_micro = int( w * 0.001 /dt )

        # A_t_pops = [moving_average(A, w) for A in A_t_pops]
        # new_A = np.mean(A_t_pops, axis=0)
        # var_A = np.var(A_t_pops, axis=0)
        for m in range(M):
            new_A = moving_average(A_t_pop[:,m], w_micro, 'ma') # use (last) one trial to plot

            (plots["p"],) = ax1.plot(
                ts_P[begin_idx:],
                new_A[begin_idx:],
                # "--k",
                label=r"$A$"+f"_{m}, micro",
            )
        # (plots['est'],) = ax1.plot(
        #     ts_P[begin_P_idx + w // 2 - 1 : -w // 2],
        #     Aest_from_sampled_spikes[begin_P_idx + w // 2 - 1 : -w // 2], 
        #     '--b', label='est')

        ax1.set_xlim(ts_P[begin_idx], ts_P[-1])
        ax1.set_xlabel('t [s]')
        ax1.set_ylabel(r'$A_{t}$ [Hz]')
        if ylim:
            ax1.set_ylim(ylim[0], ylim[1])
        # ax1.legend(handles=plots.values(), loc=loc)
        plt.setp(ax0.get_xticklabels(), visible=False)

        '''
            Dominant interval distribution
        '''
        # pop_dom_itv = [[], []]
        # for pop in A_t_pops:
        #     pop_smoothed = moving_average(pop.T, w_micro, 'gaussian').T
        #     pop_dom = [(pop_smoothed[:,0] > pop_smoothed[:,1]).astype(np.int64), 
        #                 (pop_smoothed[:,1] > pop_smoothed[:,0]).astype(np.int64)]
        #     _npops = 2 
        #     for t in range(1, pop.shape[0]):
        #         for _pop in range(_npops):
        #             if pop_dom[_pop][t-1] > 0 and pop_dom[_pop][t] == 1:
        #                 pop_dom[_pop][t] = pop_dom[_pop][t-1] + 1

        #             if pop_dom[_pop][t-1] > 0 and pop_dom[_pop][t] == 0:
        #                 if pop_dom[_pop][t-1] * dt > 2e-3:
        #                     pop_dom_itv[_pop].append(pop_dom[_pop][t-1] * dt)

        ax2 = plt.subplot(gs[2], sharex = ax1)
        # for _pop in range(_npops):
        #     ax2.hist(pop_dom_itv[_pop], density=True, bins=np.array(range(0,11))*3, alpha=0.3, label=f'{_pop}')
        ax2.plot(ts_P[begin_idx:], I_ext_vec[begin_idx:,0])
        ax2.set_xlabel('Time (s)', fontsize=18)
        ax2.set_ylabel('I_ext', fontsize=18)

        # ax2.set_xlim([0,30])
        # ax2.set_xlabel('Dominance Time (s)', fontsize=18)
        # ax2.set_ylabel('Density', fontsize=18)
        # ax2.legend(handles=plots.values(), loc=loc)

        ax2.spines['right'].set_visible(False)
        ax2.spines['top'].set_visible(False)


        plt.xticks(fontsize=16)
        plt.yticks(fontsize=16)

        plt.tight_layout()

    '''
        plot mesoscopic estimation
    ''' 
    if (plot_meso_lif):
        
        num_plots = 1 # mesoscopic estimation
        # num_plots += 1 # mess
        num_plots += 1 # dominant interval distribution 
        height_ratios = [1, 1]

        fig = plt.figure(figsize=figsize)
        gs = gridspec.GridSpec(num_plots, 1, height_ratios=height_ratios)
        # gs.update(hspace=0.0)  # remove gaps between subplots 
        plots = dict()

        # use (last) one trial to plot
        # A_t_ASMAmeso, EA_t_ASMAmeso, mess
        A_t_ASMAmeso = A_t_ASMAmesos[-1]
        EA_t_ASMAmeso = EA_t_ASMAmesos[-1]
        mess = messs[-1]
        '''
            A(t)
        '''
        if plot_micro:
            ax_meso = plt.subplot(gs[0], sharex=ax1) 
        else:
            ax_meso = plt.subplot(gs[0]) 

        begin_meso_idx = 0
        
        for m in range(M):
            new_A = moving_average(A_t_ASMAmeso[:,m], int(w*0.001/dt_meso), kernel='ma') # use (last) one trial to plot
            new_B = moving_average(EA_t_ASMAmeso[:,m], int(w*0.001/dt_meso), kernel='ma') # use (last) one trial to plot

            (plots['EASMAmeso'],) = ax_meso.plot(
                ts_meso[begin_meso_idx:],
                new_B[begin_meso_idx:],#*dt_meso*N[m],
                # "-r",
                label=r"$\bar{A}$"+f"_{m}, meso",
            )
            # (plots['ASMAmeso'],) = ax_meso.plot(
            #     ts_meso[begin_meso_idx:],
            #     new_A[begin_meso_idx:]*dt_meso*N[m],
            #     # color="orange",
            #     label=r"$A$"+f"_{m}, meso",
            # )
            

        if ylim:
            ax_meso.set_ylim(ylim[0], ylim[1])
        ax_meso.set_xlim([0, ts_meso[-1]])
        ax_meso.set_xlabel('Time (s)', fontsize=18)
        ax_meso.set_ylabel(r'$\mathbf{n}_t$', fontsize=18)
        ax_meso.set_title(r'Simulated Mesoscopic Population Activity\\ with Estimated Model ($\{\frac{\rho^\alpha_{t_l}}{\Delta t}\}_{\alpha = 1,2,3}$)', fontsize=20)
        ax_meso.legend(handles=plots.values(), loc=loc)


        ax_meso.spines['right'].set_visible(False)
        ax_meso.spines['top'].set_visible(False)

        # '''
        #     plot mess
        # ''' 
        # ax_mess = plt.subplot(gs[1], sharex=ax_meso) 
        # for m in range(M):
        #     ax_mess.plot(ts_meso[begin_meso_idx + w // 2 - 1 : -w // 2], 
        #             (1-mess)[begin_meso_idx + w // 2 - 1 : -w // 2, m])
        # ax_mess.plot(np.ones(len(mess)), 'k--')
        # ax_mess.set_ylim((0.6, 1.4))
        # ax_mess.set_xlabel('t [s]')
        # ax_mess.set_ylabel(r'$M_t$')

        '''
            Dominant interval distribution
        '''
        # pop_dom_itv = [[], []]
        # for pop in A_t_ASMAmesos:
        #     pop_smoothed = moving_average(pop.T, w, 'gaussian').T
        #     pop_dom = [(pop_smoothed[:,0] > pop_smoothed[:,1]).astype(np.int64), 
        #                 (pop_smoothed[:,1] > pop_smoothed[:,0]).astype(np.int64)]
        #     _npops = 2 
        #     for t in range(1, pop.shape[0]):
        #         for _pop in range(_npops):
        #             if pop_dom[_pop][t-1] > 0 and pop_dom[_pop][t] == 1:
        #                 pop_dom[_pop][t] = pop_dom[_pop][t-1] + 1

        #             if pop_dom[_pop][t-1] > 0 and pop_dom[_pop][t] == 0:
        #                 if pop_dom[_pop][t-1] * dt_meso > 2e-3:
        #                     pop_dom_itv[_pop].append(pop_dom[_pop][t-1] * dt_meso)

        # ax2 = plt.subplot(gs[1]) 
        # for _pop in range(_npops):
        #     ax2.hist(pop_dom_itv[_pop], density=True, bins=np.array(range(0,11))*3, alpha=0.3, label=f'{_pop}')
        # ax2.legend()

        # ax2.set_xlim([0,30])
        # ax2.set_xlabel('Dominance Time (s)', fontsize=18)
        # ax2.set_ylabel('Density', fontsize=18)
        # ax2.legend(handles=plots.values(), loc=loc)


        ax2 = plt.subplot(gs[1], sharex = ax_meso)
        # for _pop in range(_npops):
        #     ax2.hist(pop_dom_itv[_pop], density=True, bins=np.array(range(0,11))*3, alpha=0.3, label=f'{_pop}')
        ax2.plot(ts_meso[begin_meso_idx:], I_ext_vec_meso[0, 0, begin_meso_idx:])
        ax2.set_xlabel('Time (s)', fontsize=18)
        ax2.set_ylabel('I_ext', fontsize=18)


        ax2.spines['right'].set_visible(False)
        ax2.spines['top'].set_visible(False)



        plt.xticks(fontsize=16)
        plt.yticks(fontsize=16)

        plt.tight_layout()

    if saveplot:
        prefix = 'Afinal'
        suffix = str(time.time())
        plt.savefig(os.path.join(savepath, f'{prefix}_simulate_new_trial_{suffix}.svg'))
        print(os.path.join(savepath, f'{prefix}_simulate_new_trial_{suffix}.svg'))
    if not noshow:
        plt.show()
