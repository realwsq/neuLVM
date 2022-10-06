## Environment: 

```
conda create -n neuLVM python=3.8
conda activate neuLVM
conda install pip
pip install -r requirements.txt
```

## To train neuLVM:
Example commands in `3pop_BILM.sh`. 


#### Main file for training: `train_meso.py`

#### Source files include:
```
arg_parser.py # args for trainin
src/LIFmesoCell.py # meso model of multi pops implemented with tf (for the E step optimization)
src/fast_np.py # meso model of multi pops implemented with np (FAST; for the M step optimization)
src/helper.py		
```

## Secondary files:
To simulate trials with trained parameters:

`simulate_with_trainedparams.py + activity.py 	# specify the correct file to load`

or To generate training datasets with ground truth parameters:  

`simulate_with_gtparams.py + activity.py 		# specify the correct parameters`

