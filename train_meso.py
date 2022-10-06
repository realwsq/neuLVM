
import tensorflow as tf 
from tensorflow.keras import backend as K
from scipy.optimize import minimize
import numpy as np
import itertools
import warnings

import time
import copy
import os, pickle, glob
import pdb
import matplotlib.pyplot as plt 

from arg_parser import * # always go first
from src.helper import *
from src.fast_np import fast_lossM
from src.LIFmesoCell import LIFmesoCell


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



loss_history_E = []
bigepoch_history_E = []
optimizerE = tf.keras.optimizers.Adam(learning_rate=opt['E_LR_optimal'])
def train_pass_Estep(rnn, I_ext, **kwargs):
	'''
		forward pass
	'''
	start_time = time.time()
	I_ext_in = LIFmesoCell.out_to_in(I_ext=I_ext)['I_ext']
	with tf.GradientTape() as tape:
		(_, _, Z_nll_gaussian, lnPy, lnP1_y, mess, lambda_t) = rnn(I_ext_in)
		loss_y, loss_Z = rnn.cell.neglogP_of_yZ(lnPy, lnP1_y, Z_nll_gaussian, opt['Nsampled'])

		loss_P = loss_Z+loss_y
		loss_value = loss_P 

	if kwargs['timing']:
		print("--- forward %s seconds ---" % (time.time() - start_time))

	start_time = time.time()
	bigepoch, epoch, log = kwargs['bigepoch'], kwargs['epoch'], kwargs['log']
	bigepoch_history_E.append(bigepoch)
	loss_history_E.append((loss_P, loss_y, loss_Z, loss_value))
	print(f'E: {bigepoch}_{epoch}: loss{loss_value}, loss_y{loss_y}, loss_Z{loss_Z}, mess{np.mean(np.abs(mess))}, lambda_t{np.mean(lambda_t)}')

	# whether improved or not
	newsave = False 
	if (np.argsort([l[1] for l in loss_history_E])[0] == len(loss_history_E)-1) and (np.argsort([l[2] for l in loss_history_E])[0] == len(loss_history_E)-1):
		newsave = True
		print('new save!')
		if True:
			rnn.cell.save_model(os.path.join(SAVE_DIR, f"E{kwargs['bigepoch']}_est_param"))

	# log
	if log:
		# only plot once per bigepoch
		# plotting costs time & space
		if epoch == 0:
			Z_hist_est = LIFmesoCell.in_to_out(Z_hist_est=rnn.cell.Z_hist_est)['Z_hist_est'] # (B, M, A+T)
			sampled_hist_gt = LIFmesoCell.in_to_out(sampled_hist_gt=rnn.cell.sampled_hist_gt)['sampled_hist_gt'] # (B, M, A+T, Nsampled)

			plot_hidden_activity(concate_spiketrain(sampled_hist_gt, A)[0], 
				True, concate_Iext(I_ext)[0],
				False, None,
				os.path.join(SAVE_DIR, f"E_{bigepoch}_{epoch}_{kwargs['id_']}.png"), 
				label=['ground_truth', 'estimated'],
				data=[np.expand_dims(concate_Z(kwargs['A_gt'], A).T, 0),  # (1, T, M)
					np.expand_dims(concate_Z(Z_hist_est, A).T/opt['dt'], 0), # (1, T, M)
					],
				w=int(opt['plot_act_smoothw']*0.001/opt['dt'])
				)

		# update the loss plot
		idx_thisbigepoch = np.where(np.array(bigepoch_history_E)==bigepoch)[0]
		fig, axes = plt.subplots(3,2)
		losses_label = ['loss_y', 'loss_Z', 'loss_all']
		for l in range(3):
			ax2 = axes[l][0].twinx()
			axes[l][0].plot([loss_history_E[e][l+1] for e in range(len(bigepoch_history_E))], '-k')
			ax2.plot(bigepoch_history_E, 'r')
			axes[l][0].set_ylabel(losses_label[l])
			axes[l][1].plot([loss_history_E[e][l+1] for e in idx_thisbigepoch], '-k')
		axes[2][0].set_xlabel('#epoch')
		axes[2][1].set_xlabel('#epoch in bigepoch')
		plt.tight_layout()
		plt.savefig(os.path.join(SAVE_DIR, f"Eloss_{kwargs['id_']}.png"))
		plt.close()

	if kwargs['timing']:
		print("--- log %s seconds ---" % (time.time() - start_time))

	'''
		backward pass
	'''
	start_time = time.time()
	grads = tape.gradient(loss_value, rnn.cell.trainable_variables)
	grads = [tf.where(tf.math.is_nan(grad), tf.zeros_like(grad), grad) for grad in grads]

	optimizerE.apply_gradients(zip(grads[rnn.cell.trainable_parameter_num:rnn.cell.trainable_parameter_num+rnn.cell.trainable_hidden_state_num], 
				rnn.cell.trainable_variables[rnn.cell.trainable_parameter_num:rnn.cell.trainable_parameter_num+rnn.cell.trainable_hidden_state_num]))
	if kwargs['timing']:
		print("--- backward %s seconds ---" % (time.time() - start_time))

	return loss_value, newsave
	
# Since we optimize parameters using scipy.minimize (not using tf)
# we first read out current parameters
# optimize with scipy.minimize
# then update the parameters in LIFmesoCell with the ones just learned
def fasttrain_Mstep(rnn, I_ext_M, **kwargs):

	'''
		read out parameters in LIFmesoCell
	'''
	_params = LIFmesoCell.in_to_out(M=rnn.cell.M, N=rnn.cell.N, dt=rnn.cell.dt, 
				A=rnn.cell.A, T=rnn.cell.T,
				Z_hist_est=rnn.cell.Z_hist_est, init_A0=rnn.cell.init_A0, sampled_hist_gt=rnn.cell.sampled_hist_gt,
				ref_bins=rnn.cell.ref_bins,
				asyn=rnn.cell.asyn, eps=rnn.cell.eps, conmat=rnn.cell.conmat,
				amem=rnn.cell.amem, J=rnn.cell.J, rp=rnn.cell.rp, ft=rnn.cell.ft)

	A_M = int(_params['A'])
	T_perB_M = int(_params['T'])
	ref_bins_M = int(_params['ref_bins'])
	Z_hist_est_M = _params['Z_hist_est']

	'''
		optimize the parameters to minimize the loss
	'''
	_args = (opt['M'], opt['N'], opt['dt'], 
		A_M, T_perB_M, 
		Z_hist_est_M, 
		_params['init_A0'], 
		_params['sampled_hist_gt'],
		ref_bins_M,_params['asyn'], _params['eps'], _params['conmat'],
		I_ext_M,
		opt['log_mode'],
		opt['lambda_t'],
		opt['syn_delay'],
		)


	if opt['J_parameterize'] == 3:
		x0 = _params['amem'] + [_params['J'][0],_params['J'][4],_params['J'][8]]+_params['rp'] + _params['ft']
	elif opt['J_parameterize'] == 9:
		x0 = _params['amem'] + _params['J'] + _params['rp'] + _params['ft']

	bounds = [(_amem*opt['alpha'], _amem*opt['beta']) for _amem in opt['amem']] + \
			[(_J*opt['alpha'], _J*opt['beta']) for _J in opt['J']] + \
			[(_rp*opt['alpha'], _rp*opt['beta']) for _rp in opt['rp']] + \
			[(_ft*opt['alpha'], _ft*opt['beta']) for _ft in opt['ft']] 

	global Nfeval, log_X, log_loss
	Nfeval = 0
	log_X = []
	log_loss = []
	log_loss_every = 20
	def callbackF(Xi):
		global Nfeval, log_X, log_loss
		print(f'{Nfeval}:\n {Xi}')
		log_X.append(Xi)
		if Nfeval % log_loss_every == 0:
			loss = fast_lossM(Xi, *_args)
			log_loss.append(loss)
			print(loss) 
		pickle.dump({'log_X':log_X,
				'log_loss_every': log_loss_every, 
				'log_loss':log_loss,
				'Z_hist_est':_params['Z_hist_est']},
				open(os.path.join(SAVE_DIR, 'minimizor_result'),'wb'))
		Nfeval += 1
	# callbackF(np.array(x0))

	if kwargs['timing']:
		start_time = time.time()

	optimized = minimize(fast_lossM, x0, args=_args, 
			method=opt['method'],
			bounds=bounds,
			callback=callbackF,
			options={'return_all': True,
					'disp': True,
					'maxiter':opt['epochM']
					})
	if kwargs['timing']:
		print("--- %d iter M %s seconds ---" % (optimized.nit,time.time() - start_time))
		print("--- Learned parameters ---")
		print(optimized['x'][-1])


	'''
		update the parameters in LIFmesoCell with the ones just learned
	'''
	optimized_res_out = fold_variables_withorder(optimized['x'], opt['M'], dict)
	optimized_res_in = LIFmesoCell.out_to_in(**optimized_res_out,conmat=_params['conmat'])
	rnn.cell.amem.assign(tf.convert_to_tensor(optimized_res_in['amem']))
	rnn.cell.rp.assign(tf.convert_to_tensor(optimized_res_in['rp']))
	rnn.cell.ft.assign(tf.convert_to_tensor(optimized_res_in['ft']))
	rnn.cell.J.assign(tf.convert_to_tensor(optimized_res_in['J']))


	pickle.dump({'log_X':log_X,
					'log_loss_every': log_loss_every, 
					'log_loss':log_loss,
					'Z_hist_est':_params['Z_hist_est'],
					'loss_last': optimized['fun'],
					'smooth_w': opt['sampled_spike_history_smoothed_w']},
					open(os.path.join(SAVE_DIR, f"minimizor_result_{kwargs['bigepoch']}"),'wb'))



	improved = len(log_X)>1
	return optimized['fun'], improved


def train_pass(rnn, I_ext, **kwargs):
	# We estimate the latent activity with (multiple steps of) gradient descent
	# One call of func: train_pass_Estep is one step of gradient descent
	def opt_with_earlystop(func, model, input, 
						max_epochs, require_improvement,
						**kwargs):	
		best_cost=1000000
		stop = False
		last_improvement=0
		epoch = 0

		while epoch < max_epochs and stop == False:
			loss_value, newsave = func(model, input, 
						log=(epoch%opt['LOG_PER']==0), epoch=epoch, 
						**kwargs)
			if newsave:
				best_cost = loss_value
				last_improvement = 0
			else:
				last_improvement +=1
			if last_improvement >= require_improvement:
				stop = True
			epoch += 1

		# GO back to last Z_hist_est with lowest loss
		stop = False 
		epochi = kwargs['bigepoch']
		while not stop:			
			if os.path.isfile(f"{SAVE_DIR}/E{epochi}_est_param"):
				stop = True 
				Z_hist_est = pickle.load(open(f"{SAVE_DIR}/E{epochi}_est_param", 'rb'))['Z_hist_est']
			else:
				epochi -= 1
		Z_hist_est = LIFmesoCell.out_to_in(Z_hist_est=Z_hist_est)['Z_hist_est']
		model.cell.Z_hist_est.assign(tf.convert_to_tensor(Z_hist_est))	

		improved = (epochi == kwargs['bigepoch']) and (epoch > require_improvement+1)
		return None, improved 


	# We estimate the parameters with scipy.optimize.minimize
	# stopping criteria is already interior specified
	M_loss, M_improved = fasttrain_Mstep(rnn, I_ext, **kwargs)

	if opt['epochE'] > 0:
		E_loss, E_improved = opt_with_earlystop(train_pass_Estep, rnn, I_ext, 
			opt['epochE'], opt['E_patience_optimal'], 
			**kwargs)
	else:
		E_loss = None
		E_improved = False 

	improved = (M_improved and E_improved)
	print(E_improved, M_improved, improved)
	return rnn, improved



def train(epochs, suffix):
	id_ = f"{int(opt['J'][0])}_{suffix}"

	'''
		read data
			# A_gt [Btotal, M,A+T]
			# sampled_hist_gt [Btotal, M, A+T, opt['Nsampled']]
			# I_ext [Btotal, M, T]
	'''
	A_gt, sampled_hist_gt, I_ext = preprocess_gt_activity(opt['N'][0], opt['J'][0], opt['dt'])
	

	'''
		empirical estimate of Z = n/N
	'''
	if opt['sampled_spike_history_smoothed_w'] > 0:
		Z_hist_est_init = moving_average(sampled_hist_gt.mean(-1), int(opt['sampled_spike_history_smoothed_w']*0.001/opt['dt']), kernel=opt['sampled_spike_history_smoothed_kernel'])
	else:
		Z_hist_est_init = A_gt*opt['dt']

	plot_hidden_activity(concate_spiketrain(sampled_hist_gt, A)[0], 
		False, None,
		False, None,
		os.path.join(SAVE_DIR, f"init_{id_}.png"), 
		label=['ground_truth', 'estimated'],
		data=[np.expand_dims(concate_Z(A_gt, A).T, 0),  # (1, T, M)
			np.expand_dims(concate_Z(Z_hist_est_init, A).T/opt['dt'], 0), # (1, T, M)
			],
		w=int(opt['plot_act_smoothw']*0.001/opt['dt'])
		)

	'''
		EM estimate
	'''
	opt['init_amem'] = [np.random.uniform(_amem*opt['alpha'], _amem*opt['beta']) for _amem in opt['amem']]
	opt['init_J'] = [np.random.uniform(_J*opt['alpha'], _J*opt['beta']) for _J in opt['J']]
	opt['init_rp'] = [np.random.uniform(_rp*opt['alpha'], _rp*opt['beta']) for _rp in opt['rp']]
	opt['init_ft'] = [np.random.uniform(_ft*opt['alpha'], _ft*opt['beta']) for _ft in opt['ft']]

	rnn = init_model(sampled_hist_gt, A_gt*opt['dt'])
	rnn.cell.save_model(os.path.join(SAVE_DIR, f'init_param_{id_}'))

	overall_time = time.time()
	for e in range(epochs):
		rnn, improved = train_pass(rnn, I_ext,
					bigepoch=e, timing=True, A_gt=A_gt, id_=id_) 

		if not improved:
			epochs = e+1
			break

	overall_time = time.time() - overall_time
	print("============ in total training takes %s seconds ============" % (overall_time))
	print(f"============ in total training takes {epochs} epochs ============")
		

if __name__ == "__main__":

	if not os.path.exists(SAVE_DIR):
		# Create a new directory because it does not exist 
		os.makedirs(SAVE_DIR)

	suffix = str(time.time())
	train(opt['max_epochs'], suffix)


