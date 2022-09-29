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
LIFmesoCell.py # meso model of multi pops implemented with tf (for the E step optimization)
fast_np.py # meso model of multi pops implemented with np (FAST; for the M step optimization)
helper.py		
```

## Secondary files:
To simulate trials with trained parameters:

`simulate.py + activity.py 	# with load_from_trained_model = True and specify the correct file to load`

or To generate training datasets with ground truth parameters:  

`simulate.py + activity.py 	# with load_from_trained_model = False and specify the correct parameters`

