import tensorflow as tf
import numpy as np
import pickle



class LIFmesoCell(tf.keras.layers.Layer):
    def __init__(self, A, dt, 
                amem, asyn, rp, J, conmat, eps, ft, ref_t, syn_delay,
                M, N, Nsampled, T, B,
                Z_hist_est, sampled_hist_gt,
                initialize='stationary',
                simulation_mode=False,
                **kwargs):
        '''
        Z_hist_est: (B, self.M, self.A+self.T_num) nparray, float32
        sampled_hist_gt: (B, self.A+self.T_num, self.M, Nsampled) nparray, float32
        N: (self.M)
        amem, asyn, rp, ft: (self.M, 1), float32
        eps, ref_t: float64 # shared across M pops
        J: (self.M) float32 
        conmat: (self.M, self.M) float32 # J@conmat[a, b] from pop a to pop b
        '''
        # general settings
        self.dt = dt
        self.da = dt
        self.A = A
        self.a_grid = tf.linspace(0., self.A*self.da, A+1)[:-1]
        self.M = M # number of pops
        self.N = N
        self.B = B
        self.Nsampled = Nsampled
        self.T = T
        self.simulation_mode = simulation_mode

        # firing f related
        self._ft = ft

        # post-spike kernel
        self._amem = amem
        self._rp = rp
        self.ref_t = ref_t  # shared across M pops
        self.ref_bins = int(np.floor(self.ref_t/self.da))
        self.syn_delay = syn_delay
        self.syn_delay_bins = int(np.floor(self.syn_delay/self.da))

        # input kernel
        self._asyn = asyn
        self._J = J 
        self.conmat = tf.convert_to_tensor(conmat) # (conmat * diag(J))[a, b] from pop a to pop b
        self._eps = eps # shared across M pops

        self._Z_hist_est = Z_hist_est # (b,M,A+T) nparray
        self.sampled_hist_gt = sampled_hist_gt # (b,A+T, M, Nsampled) nparray
        self.init_sampled_neuron_lastfire = tf.convert_to_tensor([self.da * np.argmax(self.sampled_hist_gt[b,:self.A,:][::-1], axis=0) for b in range(B)], 
                                                tf.dtypes.float32) # (b, M, Nsampled)
        self.init_A0 = tf.convert_to_tensor(self.sampled_hist_gt[:,:self.A,:,:].mean((1,3))/self.dt, 
                                                tf.dtypes.float32)  # (b, M)
        
        self.initialize = initialize
        self.internal_clock = 0
        self._init_step = True
        self.state_size = [
                        tf.TensorShape([self.B,self.M, self.A]), # V_t_old
                        tf.TensorShape([self.B,self.M, self.A]), # S_t_old
                        tf.TensorShape([self.B,self.M, self.Nsampled]), # sampled_neuron_v
                        tf.TensorShape([self.B,self.M, self.Nsampled]), # sampled_neuron_lastfire
                        tf.TensorShape([self.B,self.M, 1]), #I_syn
                        ]

        # self.SAVE_DIR = kwargs['SAVE_DIR']
        super(LIFmesoCell, self).__init__()

    def build(self, input_shapes):
        self.amem = self.add_weight(
            shape=(self.M, 1), 
            initializer = tf.constant_initializer(self._amem), # alpha_mem = 1/tau_mem
            constraint=tf.keras.constraints.NonNeg(),
            trainable=False, name="amem",
        )
        self.asyn = self.add_weight(
            shape=(self.M, 1), 
            initializer = tf.constant_initializer(self._asyn), # alpha_mem = 1/tau_mem
            constraint=tf.keras.constraints.NonNeg(),
            trainable=False, name="asyn",
        )
        self.J = self.add_weight(
            shape=(self.M,self.M), 
            initializer = tf.constant_initializer(self._J), 
            trainable=False, name="J",
        )
        self.rp = self.add_weight(
            shape=(self.M, 1), 
            initializer = tf.constant_initializer(self._rp), 
            constraint=tf.keras.constraints.NonNeg(),
            trainable=False, name="rp",
        )
        self.ft = self.add_weight(
            shape=(self.M, 1), 
            initializer = tf.constant_initializer(self._ft), 
            trainable=False, name="ft",
        )
        self.eps = self.add_weight(
            shape=(1), 
            initializer = tf.constant_initializer(self._eps), 
            constraint=tf.keras.constraints.NonNeg(),
            trainable=False, name="eps",
        )
        self.Z_hist_est = self.add_weight(
            shape=(self.B, self.M, self.A+self.T), 
            initializer = tf.constant_initializer(self._Z_hist_est), 
            constraint=lambda z: tf.clip_by_value(z, 1e-5, 1.),
            trainable=True, name="Z_hist_est",
        )
        

        super().build(input_shapes)


    # init h_t, m_t, S_t
    def get_initial_state(self, inputs=None, batch_size=None, dtype=None):
        
        temp_init_S_t = np.ones((self.B, self.M, self.A), dtype=np.float32)
        temp_init_V_t = np.zeros((self.B, self.M, self.A), dtype=np.float32)
        temp = self.Z_hist_est.numpy().copy()
        I_syn = np.zeros((self.B, self.M, 1), dtype=np.float32)
        temp_init_sampled_neuron_v = np.zeros((self.B, self.M, self.Nsampled), dtype=np.float32)

        if self.initialize == 'stationary':
            # stationary initialization
            for m in range(self.M):
                rp = self.rp[m, 0].numpy()
                ft = self.ft[m, 0].numpy()
                am = self.amem[m,0].numpy()
                h = (rp+(self.init_A0 @ self.J)[:, m]/am).numpy()
                
                A0 = self.init_A0[:, m].numpy()
                temp_init_S_t[:,m] = np.array([[LIFmesoCell.S0(s, h[b], am, ft, refractory_t=self.ref_t, dr=1e-3) for s in self.a_grid.numpy()] for b in range(self.B)], dtype=np.float32)
                temp_init_V_t[:,m] = [LIFmesoCell.V_lastfire_rbefore(h[b], self.a_grid.numpy(), am, refractory_t=self.ref_t) for b in range(self.B)]
                temp_init_sampled_neuron_v[:,m] = [LIFmesoCell.V_lastfire_rbefore(h[b], self.init_sampled_neuron_lastfire.numpy()[b,m], am, refractory_t=self.ref_t) for b in range(self.B)]
                temp[:,m, :self.A] = A0[:,np.newaxis]*self.dt
                I_syn[:,m,0] = A0

        elif self.initialize == 'synchronized':
            # All-fire initialization
            temp[:,:,:self.A-1] = float(0.)
            temp[:,:,self.A-1] = float(1.)

        else:
            assert False 

        initialized_state = (
                             tf.convert_to_tensor(temp_init_V_t), 
                             tf.math.log(temp_init_S_t),
                             tf.convert_to_tensor(temp_init_sampled_neuron_v),
                             self.init_sampled_neuron_lastfire,
                             tf.convert_to_tensor(I_syn),
                            )
        self.Z_hist_est.assign(temp)

        self.reset()

        self.compute_shared_values()

        return initialized_state

    def compute_shared_values(self):
        self._exp_term_syn = tf.math.exp(-self.dt*self.asyn) 

    def reset(self):
        self.internal_clock = 0
        self._init_step = True 

    @tf.function
    def exp(self, x):
        return tf.math.exp(x)
        # return tf.math.exp(tf.clip_by_value(x, clip_value_min=float(-10.), clip_value_max=float(20.)))

    @tf.function
    def compute_firing_intensity(self, V):
        f = self.exp(V-self.ft) 
        f = tf.where(V==0., 1e-8, f)  # hard reset to 0 for bins in refractoryness
        return f

    @tf.function
    def Gaussian_nll_with_Z(self, Z, Z_pred):
        '''
        Z, Z_pred: [batch_size, M, 1]

        return: [batch_size, M, 1]
        '''
        mu = Z
        var = Z / self.N+ 0.1* self.dt / self.N

        mse = 0.5*tf.math.square(Z_pred-mu)/var 
        sigma_trace = 0.5*tf.math.log(var)
        log2pi = 0.5*tf.math.log(2*3.14)

        return mse + sigma_trace + log2pi

    
    @tf.function
    def get_Plam_t(self, input_total, V_t_old, age_old):
        '''
        input_total: [batch_size, M, 1]
        V_t_old: [batch_size, M, A]
        age_old: [x, x, A]
        return:
        V_t_new, Plam, lambda_telda: [batch_size, M, A]

        A or Nneurons
        '''
        # dZ (valentin formulation)= Zt (my implementation) ~= Ada
        # dZ/dt (valentin formulation) = Zt/da (my implementation) ~= A

        # # da/tau_m
        V_t_new = V_t_old + self.da * (self.rp - V_t_old) * self.amem + self.da * input_total
        V_t_new = tf.where(age_old < self.ref_t, 0., V_t_new)  # hard reset to Vreset for bins in refractoryness

        lambda_telda = self.compute_firing_intensity(V_t_new) # (b,M,a)
        Plam = lambda_telda * self.dt
        Plam = tf.where(Plam >= 0.01, 1-tf.math.exp(-Plam), Plam)

        return V_t_new, Plam, lambda_telda


    @tf.function
    def get_A_t_and_Z_t_and_Z_t_likelihood(self, logS_t_old, Plam_t, Z_t, Z_t_new):
        '''
        Z_t_new: [batch_size, M, 1]
        logS_t_old, Plam_t, Z_t: [batch_size, M, A]

        return:
        A: [batch_size, M, 1]
        Z_likelihood: [batch_size, M, 1]
        Z_new_est: [batch_size, M, 1]
        lambda_t: [batch_size, M, 1]
        mess: [batch_size, M, 1]
        '''
        S_t_old = self.exp(logS_t_old)
        m_t = S_t_old * Z_t

        Zmacro = tf.math.reduce_sum(m_t * Plam_t, axis=2, keepdims=True) # (b,M,1)

        mess = 1-tf.math.reduce_sum(m_t, axis=2, keepdims=True)
        v_t = (1-S_t_old) * m_t
        lambda_t = tf.math.reduce_sum(v_t * Plam_t, axis=2, keepdims=True)/tf.math.reduce_sum(v_t, axis=2, keepdims=True)
        # NaN: step=1
        lambda_t = tf.where(tf.math.is_nan(lambda_t), 0., lambda_t)
        lambda_t = lambda_t 

        Z = Zmacro + lambda_t * mess
        Z = tf.clip_by_value(Z, clip_value_min=0., clip_value_max=1)

        # Gaussian
        Z_new_est = Z
        if self.simulation_mode:
            Z_new_est =  tf.random.normal((self.B, self.M,1), mean=Z, stddev=tf.sqrt(Z/self.N), dtype=tf.dtypes.float32) 
            Z_new_est = tf.where(Z_new_est<0, 0., Z_new_est)

        A = Z / self.dt 

        Z_nll_gaussian = self.Gaussian_nll_with_Z(Z, Z_t_new)


        return A, Z_new_est, Z_nll_gaussian, lambda_t, mess

    @tf.function
    def get_y_t_likelihood(self, sampled_neuron_v, sampled_neuron_lastfire, input_total):
        sampled_neuron_v_new, Plam, lambda_telda = self.get_Plam_t(input_total, sampled_neuron_v, sampled_neuron_lastfire)

        _lambda_teldadt = lambda_telda * self.dt
        lnPy = tf.where(_lambda_teldadt<0.01, 
                        sampled_neuron_v_new-self.ft+tf.math.log(self.dt), 
                        tf.math.log(Plam))
        lnP1_y = -_lambda_teldadt

        return lnPy, lnP1_y, sampled_neuron_v_new

    def call(self, inputs, states):
        x_fixed = inputs[:,:] 
        V_t_old, logS_t_old, sampled_neuron_v, sampled_neuron_lastfire, I_syn = states


        if self.internal_clock > self.syn_delay_bins:
            Z_t_last = self.Z_hist_est[:,:,self.internal_clock+self.A-self.syn_delay_bins-1:self.internal_clock+self.A-self.syn_delay_bins] # (batch_size, M, 1)
            I_syn = self._exp_term_syn * I_syn + (1-self._exp_term_syn) * Z_t_last/self.dt # (M,1)*(b,M,1) + (M,1)*(b,M,1)
        else:
            pass 

        input_total = tf.expand_dims(tf.tensordot(tf.squeeze(I_syn, axis=2), 
                                                self.J,
                                                axes=[[1],[0]]), 
                                    axis=2) + \
                        tf.expand_dims(x_fixed * self.eps, axis=2) # ((b, M) @ (M, M), 1) + (b,M,1)
        '''
            update of population activity
        '''
        Z_t = self.Z_hist_est[:,:,self.internal_clock:self.internal_clock+self.A][:,:,::-1]
        Z_t_new = self.Z_hist_est[:,:,self.internal_clock+self.A:self.internal_clock+self.A+1] # (batch_size, M, 1)
        V_t_new, Plam, lambda_telda = self.get_Plam_t(input_total, V_t_old, self.a_grid)        

        A_t, Z_t_new_est, Z_t_nll_gaussian, lambda_t, mess = \
            self.get_A_t_and_Z_t_and_Z_t_likelihood(logS_t_old, Plam, Z_t, Z_t_new)
        if self.simulation_mode:
            Z_t_new = Z_t_new_est
            temp = self.Z_hist_est.numpy().copy()
            temp[:,:,self.internal_clock+self.A:self.internal_clock+self.A+1] = Z_t_new_est
            self.Z_hist_est.assign(temp)

        V_t_new = tf.concat((tf.zeros((self.B, self.M, 1)), V_t_new[:,:,:-1]), axis=2)
        # (1,M,A)

        logS_t_new = tf.concat(
                    (tf.zeros((self.B,self.M,1)), (logS_t_old-lambda_telda*self.dt)[:,:,:-1]),
                    axis=2) 


        '''
            update of sampled neuron activity
        '''
        lnPy, lnP1_y, sampled_neuron_v = self.get_y_t_likelihood(sampled_neuron_v, sampled_neuron_lastfire, input_total)
        if self.simulation_mode:
            self.sampled_hist_gt[:,self.internal_clock+self.A,:,:] = \
                (np.random.rand(self.B, self.M, self.Nsampled)<tf.math.exp(lnPy)).numpy().astype(np.float32)

        Z_t_sn_new = self.sampled_hist_gt[:,self.internal_clock+self.A,:,:]
        sampled_neuron_lastfire = tf.where(Z_t_sn_new, 0, sampled_neuron_lastfire+self.da)
        sampled_neuron_v = tf.where(Z_t_sn_new, 0, sampled_neuron_v)


        output = (A_t,                  # (b, T, M, 1)
                    Z_t_new_est,        # (b, T, M, 1)
                    Z_t_nll_gaussian,   # (b, T, M, 1)
                    lnPy, lnP1_y,       # (b, T, M, N)
                    )
        
        new_states = (V_t_new, logS_t_new, sampled_neuron_v, sampled_neuron_lastfire, I_syn)


        if self._init_step:
            self._init_step = False
        else: 
            self.internal_clock+=1

        return output, new_states


    @staticmethod
    def f(Vs, ft):
        f = np.exp(Vs-ft)
        
        if type(f) == np.ndarray or type(f) == list:
            f[Vs==0] = 0.
        elif type(f) == tf.python.framework.ops.EagerTensor:
            f = tf.where(Vs == 0, 0., f)
        else:
            if Vs == 0:
                f = 0.
        return f

    @staticmethod
    def V_lastfire_rbefore(h, rs, am, refractory_t=0.):
        V = h*(1-np.exp(-rs*am))
        
        if type(V) == np.ndarray or type(V) == list:
            V[rs<refractory_t] = 0.
        elif type(V) == tf.python.framework.ops.EagerTensor:
            V = tf.where(rs < refractory_t, 0., V)
        else:
            if rs < refractory_t:
                V = 0.
        return V

    @staticmethod
    def S0(s, h, am, ft, dr=0.00005, refractory_t=0.):
        # fire at 0, probability of survival at time s
        if s <= 0:
            return 0
        fs = LIFmesoCell.f(LIFmesoCell.V_lastfire_rbefore(h, \
                                            np.linspace(0, s, int(s/dr)+1), 
                                            am,
                                            refractory_t=refractory_t,
                                            ), 
                            ft)
        return np.exp(-np.sum(fs*dr))

    
    @tf.function
    def neglogP_of_yZ(self, lnPy, lnP1_y, Z_t_nll_gaussian, Nsampled):
        # lnPy, lnP1_y: (b, T, M, Nsampled)
        # Z_t_nll_gaussian: (b, T, M, 1)
        # self.sampled_hist_gt: (b, T, M, 1)

        # train for p(y,Z| theta)
        loss_y = -tf.reduce_mean(self.sampled_hist_gt[:,self.A:] * lnPy + (1-self.sampled_hist_gt[:,self.A:]) * lnP1_y)
        loss_Z = tf.reduce_mean(Z_t_nll_gaussian)

        return loss_y * Nsampled, loss_Z

    def save_model(self, savename):
        f = open(savename, 'wb')
        to_save = {}
        for v in self.trainable_variables:
            to_save[v.name[:-2]] = v  #v.name[-2:] = ':0'
        pickle.dump(LIFmesoCell.in_to_out(**to_save), open(savename, 'wb'))
        f.close()


    @staticmethod
    def load_model(amem, J, rp,ft, Z_hist_est,
                    asyn,eps, ref_t, conmat, syn_delay, 
                    dt,A,M, N, Nsampled, T, B,
                    sampled_hist_gt, 
                    initialize,
                    simulation_mode,
                    **useless
                   ):
        '''
        communicate with the outside: 
            take the parameters AS IT IS from the outside,
            convert,
            and set up the model
        '''


        cell = LIFmesoCell(\
                    **LIFmesoCell.out_to_in(amem=amem, J=J, rp=rp, ft=ft, Z_hist_est=Z_hist_est,
                            asyn=asyn, eps=eps, ref_t=ref_t, conmat=conmat,
                            syn_delay=syn_delay, dt=dt, A=A,
                            M=M, N=N, Nsampled=Nsampled, T=T, B=B,
                            sampled_hist_gt=sampled_hist_gt,
                            ), 
                    initialize=initialize,
                    simulation_mode=simulation_mode,
                    )


        cell.build([None, 1])
        rnn = tf.keras.layers.RNN(cell, return_sequences=True)

        return rnn

    @staticmethod
    def in_to_out(**kwargs):
        '''
        convert data from inside (parameters used inside the network) to be compatible with the outside 
        '''
        _vars = kwargs.keys()
        
        for _v in ['amem', 'asyn', 'rp', 'ft', 'lambda_t']:
            if _v in _vars:
                kwargs[_v] = kwargs[_v][:,0].numpy().astype(np.float64).tolist()

        for _v in ['conmat']:
            if _v in _vars:
                kwargs[_v] = kwargs[_v].numpy().astype(np.float64).tolist()

        # (M, M) -> (M*M)
        if 'J' in _vars:
            kwargs['J'] = kwargs['J'].numpy().astype(np.float64)
            kwargs['J'][2,:] = -kwargs['J'][2,:]  # -19 -> 19
            kwargs['J'] = kwargs['J'].reshape((-1)).tolist()

        for _v in ['ref_t', 'dt']:
            pass

        for _v in ['A', 'M', 'Nsampled', 'T', 'B', 'ref_bins']:
            pass        

        if 'eps' in _vars:
            # (1) -> double
            kwargs['eps'] = kwargs['eps'][0].numpy().astype(np.float64)

        for _v in ['N']:
            if _v in _vars:
                kwargs[_v] = kwargs[_v][:,0]

        for _v in ['Z_hist_est', 'init_A0']: 
            if _v in _vars:
                # (b, M, T) -> (b,M,T)
                # (b, M) -> (b, M)
                kwargs[_v] = kwargs[_v].numpy().astype(np.float64)

        for _v in ['sampled_hist_gt']: 
            if _v in _vars:
                # (b, A+T, M, Nsampled) -> (b,M,A+T, Nsampled)
                kwargs[_v] = np.moveaxis(kwargs[_v], 1, 2).astype(np.float64)

        for _v in ['lnPy', 'lnP1_y']: 
            if _v in _vars:
                # (b, T, M, Nsampled) -> (b,M, T, Nsampled)
                kwargs[_v] = np.moveaxis(kwargs[_v].numpy(), 1, 2).astype(np.float64)

        for _v in ['Z_nll_gaussian', 'EA_est', 'Z_est', 'mess']: 
            if _v in _vars:
                # (b, T, M, 1) -> (b, M, T)
                kwargs[_v] = np.moveaxis(kwargs[_v][:,:,:,0].numpy(), 1, 2).astype(np.float64)

        return kwargs


    @staticmethod
    def out_to_in(**kwargs):
        '''
        convert data from outside to be compatible with the inside (parameters used inside the network)
        '''
        _vars = kwargs.keys()
        
        for _v in ['amem', 'asyn', 'rp', 'ft', 'lambda_t']:
            if _v in _vars:
                kwargs[_v] = np.expand_dims(kwargs[_v], 1).astype(np.float32)

        for _v in ['conmat']:
            if _v in _vars:
                kwargs[_v] = np.array(kwargs[_v]).astype(np.float32)

        # (M*M) -> (M,M)
        if 'J' in _vars:
            if len(kwargs['J']) == 9:
                _M = 3
                kwargs['J'] = np.array(kwargs['J']).astype(np.float32)
                kwargs['J'] = kwargs['J'].reshape((_M,_M))
                kwargs['J'][2,:] = -kwargs['J'][2,:]  # 19 -> -19
            elif len(kwargs['J']) == 3:
                kwargs['J'] = np.diag(kwargs['J']) @ np.array(kwargs['conmat'])
                kwargs['J'] = kwargs['J'].astype(np.float32)

        for _v in ['eps', 'ref_t', 'dt', 'syn_delay']:
            if _v in _vars:
                pass

        for _v in ['A', 'M', 'Nsampled', 'T', 'B']:
            pass        

        for _v in ['N']:
            if _v in _vars:
                kwargs[_v] = np.expand_dims(kwargs[_v], 1)

        for _v in ['I_ext']:
            if _v in _vars:
                if len(kwargs[_v].shape) == 3:
                    # (b,M,T), (b,T,M)
                    kwargs[_v] = np.moveaxis(kwargs[_v],1,2).astype(np.float32)
                elif len(kwargs[_v].shape) == 2:
                    # (b,T), (b,T,M)
                    kwargs[_v] = np.expand_dims(kwargs[_v], 2).astype(np.float32)
                elif len(kwargs[_v].shape) == 1:
                    # (T), (1,T,1)
                    kwargs[_v] = np.expand_dims(kwargs[_v], (0,2)).astype(np.float32)
        for _v in ['Z_hist_est']: 
            if _v in _vars:
                # (b,M,T), (b, M, T)
                kwargs[_v] = np.array(kwargs[_v]).astype(np.float32)

        for _v in ['sampled_hist_gt']: 
            if _v in _vars:
                # (b, M,T, Nsampled) -> (b, T, M, Nsampled) 
                kwargs[_v] = np.moveaxis(kwargs[_v],1,2).astype(np.float32)

        return kwargs   

