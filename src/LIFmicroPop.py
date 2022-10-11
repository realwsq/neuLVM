import numpy as np

from src.LIFmesoCell import LIFmesoCell


def LIFmicroPop(
    time_end,
    dt,
    alpha_mem, 
    alpha_syn, 
    SAMPLED_NEURON_NUM,
    J=[0.],
    conmat=[[1]],
    resting_potential=[0.],
    firing_thres=[0.],
    eps=1.,
    A0=[0.0],
    refractory_t=0.0,

    I_ext_vec = None,
    M=1,
    N=[500],

    initialize='synchronized',
    **useless,
):


    steps = int(time_end / dt)
    refractory_bins = int(np.floor(refractory_t / dt))
    ts = np.linspace(0, time_end, steps+1)[:-1]

    J = np.array(J)
    conmat = np.array(conmat)
    _J = np.diag(J) @ conmat
    # _J = np.array([[J[0], 0, J[1]],[0, J[0], J[1]],[-J[2],-J[2],-J[3]]])
    A0 = np.array(A0)

    I_syn = np.zeros((M))
    u = [np.zeros(n) for n in N]         # [M, N]
    A = np.zeros((steps, M))     # [step, M]

    sampled_neurons_ids = np.array([np.sort(np.random.permutation(N[m])[:SAMPLED_NEURON_NUM]) \
                            for m in range(M)]) # (M, Nsample)
    sampled_neurons_spikes = np.zeros((M, steps, SAMPLED_NEURON_NUM), dtype=np.bool)
    sampled_neurons_probs = np.zeros((M, steps, SAMPLED_NEURON_NUM))

    age_neurons = [np.zeros(n) for n in N]

    if initialize == 'stationary':
        # stationary initialization
        for m in range(M):
            am = alpha_mem[m]
            rp = resting_potential[m]
            ft = firing_thres[m]
            h = rp + (A0 @ _J)[m]/am
            s_list = np.linspace(0.0, 0.5, int(0.5 / dt)+1)
            age_distribution = np.array([A0[m]*dt*LIFmesoCell.S0(s, h, am, ft, refractory_t=refractory_t) for s in s_list])
            age_distribution /= np.sum(age_distribution)
            age_neurons_m = np.random.choice(s_list, N[m], p=age_distribution)
            age_neurons[m] = np.sort(age_neurons_m)
            u[m] = LIFmesoCell.V_lastfire_rbefore(h, age_neurons[m], am, refractory_t=refractory_t)
    elif initialize == 'synchronized':
        # all fire at step 0
        pass
    else:
        assert False 

    A[0] = [np.sum(age_neurons[m]==0) /N[m]/dt for m in range(M)]
    sampled_neurons_spikes[:,0] = [age_neurons[m][sampled_neurons_ids[m]]==0 for m in range(M)]
    if initialize == 'stationary':
        sampled_neurons_probs[:,0] = [LIFmesoCell.f(u[m][sampled_neurons_ids[m]], firing_thres[m]) for m in range(M)]
    elif initialize == 'synchronized':
        sampled_neurons_probs[:,0] = 1.

    for s in range(1, steps):
        x_fixed = I_ext_vec[s-1]

        # I_syn += dt * (-I_syn+A[s-1]) * alpha_syn
        _exp_term_syn = np.exp(-dt*np.array(alpha_syn))  #(M)
        I_syn = _exp_term_syn * I_syn + (1-_exp_term_syn) * A[s-1] #(M)


        mu = resting_potential # (M)
        input_total = I_syn @ (_J) + eps * x_fixed # (M)
        # u += dt * ((mu-u)*alpha_mem+J * I_syn) # A[s-1])
        
        for m in range(M):
            _exp_term_tau = np.exp(-dt*alpha_mem[m]) # (M)
            u[m] = (1-_exp_term_tau) * mu[m] + _exp_term_tau * u[m] + dt * input_total[m]
            u[m][age_neurons[m]<refractory_t] = 0.0 

            prob = 1-np.exp(-LIFmesoCell.f(u[m], firing_thres[m])*dt) 
            noise = np.random.rand(N[m])
            activation = prob > noise

            u[m][activation] = 0. # reseting potential

            num_activations = np.count_nonzero(activation)

            A[s, m] = 1 / N[m] * num_activations / dt

            sampled_neurons_spikes[m, s] = activation[sampled_neurons_ids[m]]
            sampled_neurons_probs[m, s] = prob[sampled_neurons_ids[m]]
            age_neurons[m] += dt
            age_neurons[m][activation] = 0

    return ts, A, sampled_neurons_spikes, sampled_neurons_probs
