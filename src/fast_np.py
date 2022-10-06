import numpy as np
from src.LIFmesoCell import LIFmesoCell
import pdb, time
from scipy.stats import norm

def compute_firing_intensity(vmem, ft):
	# input nparrays of vmem and ft
	# output firing intensity
	def exp(x):
		return np.exp(x)
		# return np.exp(np.clip(x, -10., 20.))

	f = exp(vmem - ft) 
	f[vmem==0.] = 1e-8

	return f



def fast_lossy(x, *args):
	'''
	compute loss of P(y) = sum(S*lnP+(1-S)*ln(1-P)) FAST!!
	fun(x, *args) -> float
		x: 1d array 
			amem: *M 				# alpha_mem
			J: *M 
			rp: *M 				# resting potential
			ft: *M 				# firing threshold 

		args: a tuple of the fixed parameters needed 
		to completely specify the function. 
			0 M: (1)
			1 N: (M)
			2 dt: (1)
			3 A: (1)				# a_grid_size
			4 T: (1)
			5 Z(t): (B, M, A+T)
			6 init_A0: (B, M)
			6 S(t): (B, M, A+T, N) 	# sampled spike trains
			7 ref_bins: (1)
			8 asyn: (M)           # alpha_syn
			9 eps: (1)
			10 conmat: (M, M)
			11 I_ext: (B, M, T)			# external input
			12 syn_delay: (1)
	'''
	'''
		parse inputs
	'''
	M = args[0]
	N = np.array(args[1]) # (M)
	da = dt = args[2]
	log_dt = np.log(dt)
	A = args[3] # age grid size
	T = args[4] # time steps
	Z = np.array(args[5]) # (B, M, A+T)
	S = np.array(args[7]) # (B, M, A+T, Nsampled)
	Nsampled = S.shape[-1]

	amem = np.array(x[:M])[:,np.newaxis] # (M, 1) extra one dimension is for Nsampled
	asyn = np.array(args[9])[:,np.newaxis] # (M, 1)
	exp_mem = np.exp(-dt*amem) # (M,1)
	exp_syn = np.exp(-dt*asyn) # (M,1)
	rp = np.array(x[M+M*M:M*2+M*M])[:,np.newaxis] # (M,1)
	ft = np.array(x[M*2+M*M:M*3+M*M])[:,np.newaxis] # (M,1)
	# J = np.array(x[M:M*2]) # (M) 
	# conmat = np.array(args[10]) # (M, M)
	# J_ = np.diag(J) @ conmat # (M, M) effective connectivity matrix
	J_ = np.reshape(x[M:M+M*M], (M,M))
	eps = args[10] # float/double
	ref_bins = args[8]
	ref_t = ref_bins * dt
	syn_delay = args[13]
	syn_delay_bins = int(syn_delay/dt)

	I_ext = np.array(args[12]) # (B, M, T)
	B = I_ext.shape[0]

	'''
		values to log
	'''
	lnPy = np.zeros((B, M, T, Nsampled))
	lnP1_y = np.zeros((B, M, T, Nsampled))

	'''
		forward pass
	'''
	# Initialization: vmem, age, I_syn, Z
	
	age = np.array([da * np.argmax(S[b,:,:A][:,::-1], axis=1) for b in range(B)]) # (B, M, Nsampled), nparray
	vmem = np.zeros((B, M, Nsampled))	# (B, M, Nsampled), nparray
	init_A0 = np.array(args[6])[:, :, np.newaxis] # (B, M,1)
	# init_A0 = (S[:,:,:A,:].mean((-1,-2))/dt)[:, :, np.newaxis] # (B, M,1)
	_h = rp+ (J_.T @ init_A0)/amem # (B, M, 1)
	vmem = LIFmesoCell.V_lastfire_rbefore(_h, age, amem, refractory_t=ref_t)
	I_syn = init_A0 # (B, M,1), nparray
	Z[:,:,:A] = init_A0 * da

	# forward pass
	for t in range(A, A+T):
		if t-A > syn_delay_bins:
			I_syn = exp_syn * I_syn + (1-exp_syn) * Z[:,:,t-syn_delay_bins-1:t-syn_delay_bins] / dt # (M,1)*(B,M,1) + (M,1)*(B,M,1)
		# I_syn = exp_syn * I_syn + (1-exp_syn) * Z[:,t-1:t] / dt # (M,1)*(M,1) + (M,1)*(M,1)
		input_total = J_.T @ I_syn + I_ext[:,:,t-A:t-A+1] * eps #(M, M).T @ (B, M,1) + (B,M,1)

		# update the vmem
		vmem = vmem + da * (rp - vmem) * amem + da * input_total # (B,M,Nsampled)
		# vmem = (1-exp_mem) * rp + exp_mem * vmem + da * input_total   # (M,1) * (M,1) + (M,1) * (M,Nsampled) + (M,1)
		vmem[age < ref_t] = 0.  # hard reset to Vreset for bins in refractoryness

		# compute lambda_telda and Plam
		lambda_telda = compute_firing_intensity(vmem, ft) # (B, M,Nsampled)
		Plam = lambda_telda * dt # (B,M,Nsampled)
		Plam[Plam >= 0.01] = 1-np.exp(-Plam[Plam >= 0.01]) # (B, M,A)
		# Plam = 1-np.exp(-Plam) # (B,M,Nsampled)

		# compute lnPy, lnP1_y
		_lambda_teldadt = lambda_telda * dt # (B,M,Nsampled)
		lnPy[:,:,t-A,:][_lambda_teldadt<0.01] = (vmem-ft+log_dt)[_lambda_teldadt<0.01]  # (B,M,Nsampled)
		lnPy[:,:,t-A,:][_lambda_teldadt>=0.01] = np.log(Plam)[_lambda_teldadt>=0.01] # (B,M,Nsampled)
		lnP1_y[:,:,t-A,:] = -_lambda_teldadt # (B,M,Nsampled)

		# end of this time step, update relative values:
		# 		update age
		# 		vmem (if fires)
		age += da
		age[S[:, :, t, :]==1] = 0.
		vmem[S[:, :, t, :]==1] = 0. 
	'''
		compute loss: -sum(S*lnP+(1-S)*ln(1-P))
	'''
	loss = -np.mean(S[:, :,A:,:] * lnPy + (1-S[:,:,A:,:]) * lnP1_y)

	return loss * Nsampled

def fast_lossZ(x, *args):
	'''
	compute loss of P(y) = sum(S*lnP+(1-S)*ln(1-P)) FAST!!
	fun(x, *args) -> float
		x: 1d array 
			amem: *M 				# alpha_mem
			J: *(M*M) 
			rp: *M 				# resting potential
			ft: *M 				# firing threshold 

		args: a tuple of the fixed parameters needed 
		to completely specify the function. 
			0 M: (1)
			1 N: (M)
			2 dt: (1)
			3 A: (1)				# a_grid_size
			4 T: (1)
			5 Z(t): (B, M, A+T)
			6 init_A0: (B, M)
			7 ref_bins: (1)
			8 asyn: (M)           	# alpha_syn
			9 eps: (1)
			10 conmat: (M, M)		# not used
			11 I_ext: (B, M, T)			# external input
			12 reg_mess or big_lambda: (1)
			13 syn_delay: (1)
	'''

	'''
		parse inputs
	'''
	M = args[0]
	N = np.array(args[1]) # (M)
	da = dt = args[2]
	log_dt = np.log(dt)
	A = args[3] # age grid size
	T = args[4] # time steps
	Z = np.array(args[5]) # (B, M, A+T)
	init_A0 = np.array(args[6])[:, :, np.newaxis] # (B,M,1)

	amem = np.array(x[:M])[:,np.newaxis] # (M, 1) extra one dimension is for Nsampled
	asyn = np.array(args[8])[:,np.newaxis] # (M, 1)
	exp_mem = np.exp(-dt*amem) # (M,1)
	exp_syn = np.exp(-dt*asyn) # (M,1)
	rp = np.array(x[M+M*M:M*2+M*M])[:,np.newaxis] # (M,1)
	ft = np.array(x[M*2+M*M:M*3+M*M])[:,np.newaxis] # (M,1)
	# J = np.array(x[M:M*2]) # (M) 
	# conmat = np.array(args[10]) # (M, M)
	# J_ = np.diag(J) @ conmat # (M, M) effective connectivity matrix
	J_ = np.reshape(x[M:M+M*M], (M,M))
	eps = args[9] # float/double
	ref_bins = args[7]
	ref_t = ref_bins * dt
	syn_delay = args[13]
	syn_delay_bins = int(syn_delay/dt)

	I_ext = np.array(args[11]) # (B,M,T)
	B = I_ext.shape[0]
	log2pi = 0.5*np.log(2*3.14)

	reg_mess = args[12]

	'''
		values to log
	'''
	Z_nll_gaussian = np.zeros((B, M, T))
	messes = np.zeros((B, M, T))

	'''
		forward pass
	'''
	# Initialization: vmem, logS, age, I_syn
	age = np.linspace(0., A * da, A+1)[:-1] # (A)
	age_repeat = np.tile(age, (B,M,1)) # (B,M,A)
	vmem = np.zeros((B, M, A))	# (B, M, A)
	_h = rp+ (J_.T @ init_A0)/amem # (B, M, 1)
	vmem = LIFmesoCell.V_lastfire_rbefore(_h, age_repeat, amem, refractory_t=ref_t)
	S = np.array([[[
						LIFmesoCell.S0(s, _h[b, m,0], amem[m,0], ft[m,0], refractory_t=ref_t, dr=0.001) 
					for s in age] for m in range(M)] for b in range(B)]) # (B, M, A)
	logS = np.log(S)
	I_syn = init_A0 # (B, M,1)
	Z[:, :,:A] = init_A0 * da

	# forward pass
	for t in range(A, A+T):
		if t-A > syn_delay_bins:
			I_syn = exp_syn * I_syn + (1-exp_syn) * Z[:,:,t-syn_delay_bins-1:t-syn_delay_bins] / dt # (M,1)*(B,M,1) + (M,1)*(B,M,1)
		# I_syn = exp_syn * I_syn + (1-exp_syn) * Z[:,t-1:t] / dt # (M,1)*(M,1) + (M,1)*(M,1)
		input_total = J_.T @ I_syn + I_ext[:,:,t-A:t-A+1] * eps #(M, M).T @ (B, M,1) + (B,M,1)

		# update the vmem
		vmem = vmem + da * (rp - vmem) * amem + da * input_total  # (B,M,Nsampled)
		# vmem = (1-exp_mem) * rp + exp_mem * vmem + da * input_total   # (M,1) * (M,1) + (M,1) * (M,A) + (M,1)
		vmem[age_repeat < ref_t] = 0.  # hard reset to Vreset for bins in refractoryness

		# compute lambda_telda and Plam
		lambda_telda = compute_firing_intensity(vmem, ft) # (B,M,A)
		Plam = lambda_telda * dt # (B,M,A)
		Plam[Plam >= 0.01] = 1-np.exp(-Plam[Plam >= 0.01]) # (B,M,A)
		# Plam = 1-np.exp(-Plam) # (B,M,A)

		# estimate Z
		S = np.exp(logS)   	# (B,M,A)
		m = S * Z[:,:,t-A:t][:,:,::-1] 	# (B,M,A)*(B,M,A)
		v = (1-S) * m 		# (B,M,A)*(B,M,A)

		Zmacro = np.sum(m * Plam, -1) # (B,M)

		mess = 1-np.sum(m, -1) # (B,M)
		messes[:, :, t-A] = mess
		lambda_t = np.sum(v * Plam, -1)/np.sum(v, -1)  # (B,M)
		lambda_t[np.isnan(lambda_t)] = 1e5
		# lambda_t[mess > 0.1] = 1e5
		# lambda_t = reg_mess


		Z_pred = Zmacro + lambda_t * mess # (B,M)
		Z_pred = np.clip(Z_pred, 0., 1)

		# compute lnPZ
		_mu = Z_pred
		_var = Z_pred / N + 0.1* dt / N
		mse = 0.5*(Z[:,:,t]-_mu)**2/_var 
		sigma_trace = 0.5*np.log(_var)
		Z_nll_gaussian[:,:,t-A] = mse + sigma_trace + log2pi

		# end of this time step, update relative values:
		# 		update vmem
		# 		update logS
		vmem = np.concatenate((np.zeros((B, M, 1)), vmem), -1)[:,:,:-1]
		logS = np.concatenate((np.zeros((B, M, 1)), logS-lambda_telda*dt), -1)[:,:,:-1] # (B,M,A)

	'''
		compute loss: -lnP
	'''
	loss = np.mean(Z_nll_gaussian) # + np.mean(np.abs(messes)) * reg_mess

	return loss



def fast_lossM(x, *myargs):
	'''
	compute loss of P(y) = sum(S*lnP+(1-S)*ln(1-P)) FAST!!
	fun(x, *args) -> float
		x: 1d array 
			amem: *M 					# alpha_mem
			J: *M) 
			rp: *M 						# resting potential
			ft: *M 						# firing threshold 

		args: a tuple of the fixed parameters needed 
		to completely specify the function. 
			0 M: (1)
			1 N: (M)
			2 dt: (1)
			3 A: (1)					# a_grid_size
			4 T: (1)
			5 Z(t): (B, M, A+T)
			6.1 init_A0: (B, M)
			6.2 S(t): (B, M, A+T, N) 	# sampled spike trains
			7 ref_bins: (1)
			8 asyn: (M)           		# alpha_syn
			9 eps: (1)
			10 conmat: (M, M)
			11 I_ext: (B, M, T)			# external input
			12 log_mode: (1) 			# 0 (y+Z); 1 (y); 2 (Z)
			13 reg_mess: (1)
			14 syn_delay: (1)			# synaptic delay
	'''
	if len(x) == 18:
		# J_parameterize = 9
		J = x[3:12]
		x = [x[0], x[1], x[2],
			x[3], x[4], x[5],
			x[6], x[7], x[8],
			-x[9],-x[10],-x[11],
			x[12], x[13], x[14], 
			x[15], x[16], x[17]]
	elif len(x) == 12:
		# J_parameterize = 3
		J = x[3:6]
		x = [x[0], x[1], x[2],
			x[3], 0, x[3],
			0, x[4], x[4],
			-x[5],-x[5],-x[5],
			x[6], x[7], x[8], 
			x[9], x[10], x[11]]  

	if myargs[13] == 1: # log_mode
		return fast_lossy(x, *(myargs[:13]+myargs[15:16]))
	elif myargs[13] == 0:
		loss_y = fast_lossy(x, *(myargs[:13]+myargs[15:16]))
		loss_Z = fast_lossZ(x, *(myargs[:7]+myargs[8:13]+myargs[14:16]))
		print(loss_y, loss_Z)
		return loss_Z + loss_y
	elif myargs[13] == 2:
		loss_Z = fast_lossZ(x, *(myargs[:7]+myargs[8:13]+myargs[14:16]))
		print(loss_Z)
		return loss_Z

