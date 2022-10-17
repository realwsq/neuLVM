
import tensorflow as tf 
from scipy.optimize import minimize
import numpy as np

import time, os, pickle, pdb
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
EMstep_history_E = []
optimizerE = tf.keras.optimizers.Adam(learning_rate=opt['E_LR'])
def tf_EGDstep(rnn, I_ext, **kwargs):
	'''
		forward pass
	'''
	I_ext_in = LIFmesoCell.out_to_in(I_ext=I_ext)['I_ext']
	with tf.GradientTape() as tape:
		(_, _, Z_nll_gaussian, lnPy, lnP1_y) = rnn(I_ext_in)
		loss_y, loss_Z = rnn.cell.neglogP_of_yZ(lnPy, lnP1_y, Z_nll_gaussian, opt['Nsampled'])

		loss_P = loss_Z+loss_y
		loss_value = loss_P 

	EMstep, GDstep = kwargs['EMstep'], kwargs['GDstep']
	EMstep_history_E.append(EMstep)
	loss_history_E.append((loss_P, loss_y, loss_Z, loss_value))
	
	# whether improved or not
	newsave = False 
	if (np.argsort([l[1] for l in loss_history_E])[0] == len(loss_history_E)-1) and (np.argsort([l[2] for l in loss_history_E])[0] == len(loss_history_E)-1):
		newsave = True
		rnn.cell.save_model(os.path.join(SAVE_DIR, f"E{kwargs['EMstep']}_est_param"))

	if kwargs['log']:
		print(f"E{GDstep} step of {EMstep} EM iteration: ")
		print(f"\t loss{loss_value}, loss_y{loss_y}, loss_Z{loss_Z}")
		print(f"\t newsave={newsave}")

		Z_hist_est = LIFmesoCell.in_to_out(Z_hist_est=rnn.cell.Z_hist_est)['Z_hist_est'] # (B, M, A+T)
		sampled_hist_gt = LIFmesoCell.in_to_out(sampled_hist_gt=rnn.cell.sampled_hist_gt)['sampled_hist_gt'] # (B, M, A+T, Nsampled)

		plot_hidden_activity(concatenate_spiketrain(sampled_hist_gt, A)[0], 
			True, concatenate_Iext(I_ext)[0],
			False, None,
			os.path.join(SAVE_DIR, f"E_{EMstep}_{GDstep}.png"), 
			label=['ground_truth', 'estimated'],
			data=[np.expand_dims(concatenate_Z(kwargs['A_gt'], A).T, 0),  # (1, T, M)
				np.expand_dims(concatenate_Z(Z_hist_est, A).T/opt['dt'], 0), # (1, T, M)
				],
			w=int(opt['plot_act_smoothw']*0.001/opt['dt'])
			)

		plot_loss(loss_history_E, EMstep_history_E, EMstep)


	'''
		backward pass
	'''
	grads = tape.gradient(loss_value, rnn.cell.trainable_variables)
	grads = [tf.where(tf.math.is_nan(grad), tf.zeros_like(grad), grad) for grad in grads]

	optimizerE.apply_gradients(zip(grads, 
				rnn.cell.trainable_variables))

	return loss_value, newsave
	
# Since we optimize parameters using scipy.minimize (not using tf)
# we first read out current parameters
# optimize with scipy.minimize
# then update the parameters in LIFmesoCell with the ones just learned
def scipy_Mstep(rnn, I_ext_M, **kwargs):

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
		opt['syn_delay'],
		)


	x0 = _params['amem'] + [_params['J'][0],_params['J'][4],_params['J'][8]]+_params['rp'] + _params['ft']

	bounds = [(_amem*opt['lb'], _amem*opt['ub']) for _amem in opt['amem']] + \
			[(_J*opt['lb'], _J*opt['ub']) for _J in opt['J']] + \
			[(_rp*opt['lb'], _rp*opt['ub']) for _rp in opt['rp']] + \
			[(_ft*opt['lb'], _ft*opt['ub']) for _ft in opt['ft']] 

	global Nfeval
	Nfeval = 0
	def callbackF(X):
		global Nfeval
		Nfeval += 1
		if (Nfeval % opt['LOG_PER']) == 0:
			loss = fast_lossM(X, *_args)
			print(f"M{Nfeval} step of {kwargs['EMstep']} EM iteration: ")
			print(f"\t current set of parameters ({variablenamed_withorder(opt['M'])}): {X}")
			print(f"\t current loss: {loss}") 


	optimized = minimize(fast_lossM, x0, args=_args, 
			method=opt['method'],
			bounds=bounds,
			callback=callbackF,
			options={'return_all': True,
					'maxiter':opt['Mmaxiters']
					})

	'''
		update the parameters in LIFmesoCell with the ones just learned
	'''
	optimized_res_out = fold_variables_withorder(optimized['x'], opt['M'], dict)
	optimized_res_in = LIFmesoCell.out_to_in(**optimized_res_out,conmat=_params['conmat'])
	rnn.cell.amem.assign(tf.convert_to_tensor(optimized_res_in['amem']))
	rnn.cell.rp.assign(tf.convert_to_tensor(optimized_res_in['rp']))
	rnn.cell.ft.assign(tf.convert_to_tensor(optimized_res_in['ft']))
	rnn.cell.J.assign(tf.convert_to_tensor(optimized_res_in['J']))


	pickle.dump({'X':optimized['x'],
					'Z_hist_est':_params['Z_hist_est'],
					'loss_last': optimized['fun'],},
					open(os.path.join(SAVE_DIR, f"minimizor_result_{kwargs['EMstep']}"),'wb'))

	print(f"--- Learned parameters ({variablenamed_withorder(opt['M'])})---:")
	print(optimized['x'][-1])



	improved = Nfeval>1
	return optimized['fun'], improved


def perform_EM(rnn, I_ext, **kwargs):
	# We estimate the latent activity with (multiple steps of) gradient descent
	# One call of func: tf_EGDstep is one step of gradient descent
	def opt_with_earlystop(func, model, input, 
						max_GDs, require_improvement,
						**kwargs):	
		stopGD = False
		last_improvement=0
		GDstep = 0

		while GDstep < max_GDs and stopGD == False:
			loss_value, newsave = func(model, input, 
						log=(GDstep%opt['LOG_PER']==0), GDstep=GDstep, 
						**kwargs)
			if newsave:
				last_improvement = 0
			else:
				last_improvement +=1
			if last_improvement >= require_improvement:
				stopGD = True
			GDstep += 1

		# GO back to last Z_hist_est with lowest loss
		stopGoBack = False 
		epochi = kwargs['EMstep']
		while not stopGoBack:			
			if os.path.isfile(f"{SAVE_DIR}/E{epochi}_est_param"):
				stopGoBack = True 
				Z_hist_est = pickle.load(open(f"{SAVE_DIR}/E{epochi}_est_param", 'rb'))['Z_hist_est']
			else:
				epochi -= 1
		Z_hist_est = LIFmesoCell.out_to_in(Z_hist_est=Z_hist_est)['Z_hist_est']
		model.cell.Z_hist_est.assign(tf.convert_to_tensor(Z_hist_est))	

		improved = (epochi == kwargs['EMstep']) and (GDstep > require_improvement+1)
		return None, improved 

	'''
		M step
	'''
	# We estimate the parameters with scipy.optimize.minimize
	# stopping criteria is already interiorly specified
	M_loss, M_improved = scipy_Mstep(rnn, I_ext, **kwargs)

	'''
		E step
	'''
	# We estimate the pop. act. with Adam algorithm.
	# the optimization stops when either the maximum number of Gradient Descent (opt['Emaxiters']) is reached, 
	# or the objective function stops improving for the last opt['E_itertol'] iterations. 
	# We wrap each step of gradient descent (tf_EGDstep) in opt_with_earlystop function.
	if opt['Emaxiters'] > 0:
		E_loss, E_improved = opt_with_earlystop(tf_EGDstep, rnn, I_ext, 
			opt['Emaxiters'], opt['E_itertol'], 
			**kwargs)
	else:
		E_loss = None
		E_improved = False 

	improved = (M_improved and E_improved)
	return rnn, improved



def train(max_EMsteps, A_gt, sampled_hist_gt, I_ext):
	'''
		EM estimate
	'''
	rnn = init_model(sampled_hist_gt, A_gt*opt['dt'])
	rnn.cell.save_model(os.path.join(SAVE_DIR, f'init_param'))

	overall_time = time.time()
	for e in range(max_EMsteps):
		rnn, improved = perform_EM(rnn, I_ext,
					EMstep=e, A_gt=A_gt) 

		if not improved:
			max_EMsteps = e+1
			break

	overall_time = time.time() - overall_time
	print("============ in total training takes %s seconds ============" % (overall_time))
	print(f"============ in total training takes {max_EMsteps} EMsteps ============")
		

if __name__ == "__main__":

	if not os.path.exists(SAVE_DIR):
		# Create a new directory because it does not exist 
		os.makedirs(SAVE_DIR)

	'''
		read data
			# A_gt [B, M,A+T]: ground truth pop. act. A=Z/dt=n/N/dt
			# sampled_hist_gt [B, M, A+T, opt['Nsampled']]
			# I_ext [B, M, T]
	'''
	A_gt, sampled_hist_gt, I_ext = preprocess_gt_activity(opt['N'][0], opt['J'][0], opt['dt'])
	

	'''
		randomly sample the initial parameters from the prior
	'''
	opt['init_amem'] = [np.random.uniform(_amem*opt['lb'], _amem*opt['ub']) for _amem in opt['amem']]
	opt['init_J'] = [np.random.uniform(_J*opt['lb'], _J*opt['ub']) for _J in opt['J']]
	opt['init_rp'] = [np.random.uniform(_rp*opt['lb'], _rp*opt['ub']) for _rp in opt['rp']]
	opt['init_ft'] = [np.random.uniform(_ft*opt['lb'], _ft*opt['ub']) for _ft in opt['ft']]

	train(opt['max_EMsteps'], A_gt, sampled_hist_gt, I_ext)


