import numpy as np
from scipy.ndimage import gaussian_filter1d
import matplotlib.pyplot as plt 
import os, pickle, glob


from arg_parser import *
from src.LIFmesoCell import LIFmesoCell



def init_model(sampled_hist_gt, Z_hist_gt):
	'''
	sampled_hist_gt: (Btotal, M, A+T, Nsampled)
	Z_hist_gt: (Btotal, M, A+T)
	'''
	if opt['sampled_spike_history_smoothed_w'] > 0:
		Z_hist_est = moving_average(sampled_hist_gt.mean(-1).reshape((-1, A+T_perB)), 
										int(opt['sampled_spike_history_smoothed_w']*0.001/opt['dt']), 
										kernel=opt['sampled_spike_history_smoothed_kernel']
										).reshape((Btotal, opt['M'], A+T_perB))
										# (Btotal, M, A+T)
	elif opt['sampled_spike_history_smoothed_w'] == 0:
		Z_hist_est = Z_hist_gt
	else: # < 0
		Z_hist_est = moving_average(Z_hist_gt.reshape((-1, A+T_perB)), 
										-int(opt['sampled_spike_history_smoothed_w']*0.001/opt['dt']), 
										kernel=opt['sampled_spike_history_smoothed_kernel']
										).reshape((Btotal, opt['M'], A+T_perB))


	best_rnn = LIFmesoCell.load_model(\
					opt['init_amem'], opt['init_J'], opt['init_rp'], opt['init_ft'], Z_hist_est, 
					opt['asyn'], opt['eps'], opt['ref_t'], opt['conmat'], opt['lambda_t'], opt['syn_delay'],
					opt['dt'], A, opt['M'], opt['N'], opt['Nsampled'], T_perB, Btotal,
					sampled_hist_gt,
					opt['initialize'],
					opt['inference_mode'],
					opt['update_popact'],
					opt['update_sampledneuron']
					)

	return best_rnn

def load_model_from_file(sampled_hist_gt, file_name):

	rnn = LIFmesoCell.load_model_from_savedfile(os.path.join(SAVE_DIR,file_name), 
					opt['asyn'], opt['eps'], opt['ref_t'], opt['conmat'], opt['lambda_t'], opt['syn_delay'],
					opt['dt'], A, opt['M'], opt['N'], opt['Nsampled'], TIME_STEP, BATCH_SIZE,
					sampled_hist_gt,
					opt['initialize'],
					opt['inference_mode'],
					opt['update_popact'],
					opt['update_sampledneuron'])

	return rnn

'''
utils
'''
def moving_average(x, w, kernel='ma'):
	if len(np.array(x).shape)==1:
		if kernel == 'ma':
		    return np.convolve(x, np.ones(w), 'same') / w 
		elif kernel == 'gaussian':
			return gaussian_filter1d(x, w, mode='reflect')
	if kernel == 'ma_forward':
	    return np.array([np.convolve(x[i], np.concatenate((np.zeros(w-1),np.ones(w)),axis=None), 'same') / w for i in range(len(x))])
	if kernel == 'ma':
	    return np.array([np.convolve(x[i], np.ones(w), 'same') / w for i in range(len(x))])
	elif kernel == 'gaussian':
		return np.array([gaussian_filter1d(x[i], w, mode='reflect') for i in range(len(x))])


def unfold_variables_withorder(vars):
	
	ret = []

	if type(vars) == dict:
		ret = list(vars['amem']) 
		ret += list(vars['J']) 
		ret += list(vars['rp'])  
		ret += list(vars['ft']) 
	elif type(vars) == list:
		ret = vars[0] + vars[1] + vars[2] + vars[3]
	
	return ret

def variablenamed_withorder(M):
	ret = ['tau_mem'] * M 
	if opt['J_parameterize'] == 9:
		ret += ['J'] * (M*M) 
	elif opt['J_parameterize'] == 3:
		ret += ['J'] * M 
	ret += ['rp'] * M 
	ret += ['ft'] * M 

	return ret

def fold_variables_withorder(param_list, M, totype):
	if opt['J_parameterize'] == 9:
		if totype == dict:
			ret = {}
			ret['amem'] = param_list[:M]
			ret['J'] = param_list[M:M+M*M]
			ret['rp'] = param_list[M+M*M:M*2+M*M]
			ret['ft'] = param_list[M*2+M*M:M*3+M*M]
		elif totype == list: 
			ret = [param_list[:M]]
			ret += param_list[M:M+M*M]
			ret += param_list[M+M*M:M*2+M*M]
			ret += param_list[M*2+M*M:M*3+M*M]
	elif opt['J_parameterize'] == 3:
		if totype == dict:
			ret = {}
			ret['amem'] = param_list[:M]
			ret['J'] = param_list[M:M*2]
			ret['rp'] = param_list[M*2:M*3]
			ret['ft'] = param_list[M*3:M*4]
		elif totype == list: 
			ret = [param_list[:M]]
			ret += [param_list[M:M*2]]
			ret += [param_list[M*2:M*3]]
			ret += [param_list[M*3:M*4]]
		
	return ret

def get_max_minimizor_result_from_folder(folder_name):
	# max # of foldername/minimizor_result_#
	_pb = glob_all_files_endwith_int(os.path.join(folder_name, 'minimizor_result_'))
	_pb = [int(_f.split('_')[-1]) for _f in _pb]
	best = np.max(_pb)
	return best

def glob_all_files_endwith_int(file_name_base, n=2, file_name_end=''):
	fs = []
	file_name = file_name_base
	for i in range(n):
		file_name = f"{file_name}[0-9]"
		fs = fs + glob.glob(f"{file_name}{file_name_end}")
	return fs

def concate_spiketrain(st, A):
	# st: (B,M,A+T,Nsampled)
	# return (B*T, M*Nsampled), (M,B*T,Nsampled)
	B,M,T,Nsampled = st.shape 
	T = T-A
	st = np.moveaxis(st[:,:,A:,:], 0, 1).reshape((M,-1,Nsampled)) # (M, B*T, Nsampled)

	return np.moveaxis(st, 1, 0).reshape((B*T, M*Nsampled)), st

def concate_Iext(Iext):
	# Iext: (B,M,T)
	# return (M,B*T)
	B, M, T = Iext.shape
	Iext = np.moveaxis(Iext, 0, 1)
	Iext = np.reshape(Iext, (M, B*T))
	return Iext

def concate_Z(Z, A):
	# Z: (B,M,A+T)
	B,M,T = Z.shape 
	T = T-A
	Z = np.moveaxis(Z[:,:,A:], 0, 1).reshape((M,-1)) # (M, B*T)

	return Z

def get_best_trained_files(root, multi_train_folders, ):
	best_loss = 1e9
	best_trainedfolder = best_Mfile = best_Efile = None
	for hist in multi_train_folders:

		folder = os.path.join(root, hist)
		_pb = glob.glob(os.path.join(folder, 'E[0-9]_est_param'))
		# _pb = glob.glob(os.path.join(_SAVE_DIR, 'minimizor_result_[0-9]'))
		_pb = [int(_f.split('_')[-3][-1]) for _f in _pb]
		best = np.max(_pb)
		trained_Mfile = os.path.join(folder, f'minimizor_result_{best}')
		trained_Efile = os.path.join(folder, f'E{best}_est_param')
		est_model = pickle.load(open(trained_Mfile, 'rb') )
		print(folder, best, est_model['log_loss'][-1])
		if est_model['log_loss'][-1] < best_loss:
			best_Mfile = trained_Mfile
			best_Efile = trained_Efile
			best_trainedfolder = folder
			best_loss = est_model['log_loss'][-1]
	return best_Mfile, best_Efile, best_trainedfolder

'''
plot
'''

def plot_sampled_spike_trains(sampled_spikes, savename):
	# sampled_spikes should be of shape [Nneuron, Tstep]
	fig, ax=plt.subplots(1,1,sharex=True)  # no likelihood yet

	Nneuron = sampled_spikes.shape[1]
	colors = 'black'
	ts_P = np.array(list(range(sampled_spikes.shape[0])))
	for ni in range(Nneuron):
		ax.eventplot(
				ts_P[sampled_spikes[:,ni].astype(np.bool)],
				lineoffsets=ni+1,
				colors=colors,
				# linewidths=0.5,
			)
	ax.set_ylabel('# Neuron')
	ax.set_title('Sampled Spike Trains')
			
	plt.tight_layout()
	# plt.show()
	plt.savefig(savename)
	plt.close()



def plot_hidden_activity(sampled_spikes, 
						viz_I_ext, I_ext,
						viz_ylikelihood, y_likelihood, 
						savename, **kwargs):
	# sampled_spikes should be of shape [Tstep,Nneuron]
	# y_likelihood should be of shape [1, Tstep, Nneuron]
	# kwargs: activity to plot{labelname: [N, timestep, M]}
	
	fig, ax=plt.subplots(viz_I_ext+2+viz_ylikelihood,1,sharex=True, figsize=(10,8))  # no likelihood yet

	if viz_I_ext:
		ax[0].plot(I_ext, label='I_ext')
		ax[0].legend()
		ax[0].set_ylabel('I (A)')
		ax[0].set_title('External Input')

	Nneuron = sampled_spikes.shape[1]
	colors = 'black'
	# colors1 = ['C{}'.format(i) for i in range(Nneuron)]
	ts_P = np.array(list(range(sampled_spikes.shape[0])))
	for ni in range(Nneuron):
		ax[int(viz_I_ext)].eventplot(
				ts_P[sampled_spikes[:,ni].astype(np.bool)],
				lineoffsets=ni+1,
				colors=colors,
				# linewidths=0.5,
			)
	ax[int(viz_I_ext)].set_title('Spike Trains of the Recorded Neurons', fontsize=20)
	ax[0].tick_params(
		    axis='both',          # changes apply to the x-axis
		    which='both',      # both major and minor ticks are affected
		    left=False,      # ticks along the left edge are off
		    right=False,         # ticks along the right edge are off
		    top=False,
		    bottom=False,
		    labeltop=False,
		    labelbottom=False,
		    labelleft=False) # labels along the left edge are off
	ax[0].spines['top'].set_visible(False)
	ax[0].spines['right'].set_visible(False)
	ax[0].spines['bottom'].set_visible(False)
	ax[0].spines['left'].set_visible(False)

	# ymin, ymax = 0, 70
	_plot_dyn(ax=ax[1+viz_I_ext],
			**kwargs)
	ax[1+viz_I_ext].set_ylabel('Firing Rate (Hz)', fontsize=18)
	ax[1+viz_I_ext].set_title('Population Activity (Latent Variable)', fontsize=20)
	ax[1+viz_I_ext].set_ylim([0,50])


	if viz_ylikelihood:
		_plot_dyn(ax=ax[2+viz_I_ext],
				label=['y_likelihood'],
				data=[y_likelihood])

	ax[-1].set_xlabel(f"Time step (dt={opt['dt']*1000}ms)", fontsize=18)
	ax[-1].set_xlim([0,sampled_spikes.shape[0]])

			
	plt.tight_layout()
	# plt.show()
	plt.savefig(savename)
	plt.close()



def _plot_dyn(ax=None, alpha=1, w=1, **kwargs):
	if ax is None:
		fig, ax = plt.subplots(1,1) 
	for ki in range(len(kwargs['data'])):
		k = kwargs['label'][ki]
		v = kwargs['data'][ki]
		if v is None:
			continue
		for i in range(v.shape[0]):
			smoothed_v = moving_average(v[i].T, w, kernel='gaussian')
			for j in range(smoothed_v.shape[0]):
				if 'style' in kwargs:
					ax.plot(smoothed_v[j],kwargs['style'][ki], label=k, alpha=alpha)
				else:
					ax.plot(smoothed_v[j],label=k, alpha=alpha)
	ax.legend()

	plt.xticks(fontsize=16)
	plt.yticks(fontsize=16)

	ax.spines['right'].set_visible(False)
	ax.spines['top'].set_visible(False)




def preprocess_gt_activity(N, J, dt, random_sample_neuron=False): 	
	_A = int(opt['a_cutoff']/dt)
	end_time = int(opt['end_timepoint']/dt) 
	
	# winner-take-all 3pops example
	if N == 400 and J == 9.984:
		if opt['with_input']==0:
			# info about the artificial dataset:
			# T=200s, dt=0.2ms,  
			with open('./dataset/3pop_noinput_small', 'rb') as f:
				data = pickle.load(f)
		# with random inputs 
		elif opt['with_input']==1:
			# info about the artificial dataset:
			# T=100s, dt=0.2ms,  
			with open('./dataset/3pop_Pozzorini20', 'rb') as f:
				data = pickle.load(f)		

		dt_micro = 0.0002
		assert dt/dt_micro-int(dt/dt_micro) == 0

		'''
		data with fields:
		A_t_pop: T*M
		sampled_spikes: M*T*Nsampled
		I_ext: T*M
		'''
		A_gt_raw, sampled_spikes_raw = data['A_t_pop'], data['sampled_spikes']  

		# Process1: re-bin
		A_gt_raw = A_gt_raw.reshape((-1, int(dt/dt_micro), opt['M'])).mean(axis=1); 
		sampled_spikes_raw = sampled_spikes_raw.reshape((opt['M'], -1, int(dt/dt_micro), sampled_spikes_raw.shape[2])).sum(axis=2) # > 0 
		I_ext_vec_raw = data['I_ext'].reshape((-1, int(dt/dt_micro), opt['M'])).mean(axis=1); 

		# Process2: cut out the right time window
		A_gt_raw = np.array(A_gt_raw[end_time-A-int(opt['trial_length']/dt):end_time].T, dtype=np.float64) # (M,A+T) 
		sampled_spikes_raw = sampled_spikes_raw > 0
		sampled_spikes_raw = np.array(sampled_spikes_raw[:,end_time-_A-int(opt['trial_length']/dt):end_time], dtype=np.float64) # [M, A+T, #neuron]
		I_ext_vec = np.array(I_ext_vec_raw[end_time-int(opt['trial_length']/dt):end_time, :].T, dtype=np.float64) # (M,T) 
	

	else:
		assert False
	
	# should have the following forms:
	# A_gt_raw (M,A+T）, nparray, np.float64
	# sampled_spikes_raw (M, A+T, Nsampled）, nparray, np.float64
	# I_ext_vec (M,T)， nparray, np.float64

	# Process3: 'sample' the recorded neurons
	a = np.full(sampled_spikes_raw.shape[2], False)
	a[:opt['Nsampled']] = True
	if random_sample_neuron:
		np.random.shuffle(a)
	sampled_spikes_raw = sampled_spikes_raw[:,:,a]

	# 
	if not (A_gt_raw is None):
		Z_hist_gt = A_gt_raw*dt # (M，A+T)
	else:
		Z_hist_gt = None

	# Process4: batch
	_Btotal = int(opt['trial_length']/opt['t_perbatch'])
	_T_perB = int(opt['t_perbatch']/dt)

	A_gt_inbatch = None if (A_gt_raw is None) else np.zeros((_Btotal, opt['M'], _A+_T_perB), dtype=np.float64)
	sampled_spikes_raw_inbatch = np.zeros((_Btotal, opt['M'], _A+_T_perB, opt['Nsampled']), dtype=np.float64)
	I_ext_vec_inbatch = np.zeros((_Btotal, opt['M'], _T_perB), dtype=np.float64)

	for _b in range(_Btotal):
		if not (A_gt_raw is None):
			A_gt_inbatch[_b] = A_gt_raw[:,_T_perB*_b:_T_perB*_b+_A+_T_perB]
		sampled_spikes_raw_inbatch[_b] = sampled_spikes_raw[:,_T_perB*_b:_T_perB*_b+_A+_T_perB]
		I_ext_vec_inbatch[_b] = I_ext_vec[:, _T_perB*_b:_T_perB*_b+_T_perB]

	return A_gt_inbatch, sampled_spikes_raw_inbatch, I_ext_vec_inbatch

