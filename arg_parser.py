from optparse import OptionParser
from optparse import OptionGroup
import time
import argparse, os


def parse_args():
    # load parameters and options
    parser = OptionParser()

    parser = parse_neuLVM_parameter_fitting_args(parser)

    (opt, args) = parser.parse_args()
    parser.defaults
    opt.time = time.ctime()


    return vars(opt) # to dict


def parse_neuLVM_parameter_fitting_args(parser):

    group = OptionGroup(parser, "Experiment settings")
    group.add_option(
        "--trial_length",
        type=int,
        default=10, 
        help='t (s); '
    )
    group.add_option(
        "--with_input",
        type=int,
        default=0, 
        help='with input (1) or not (0)'
    )
    group.add_option(
        "--t_perbatch",
        type="float",
        default=10, 
        help='t (s); '
    ) 
    group.add_option(
        "--end_timepoint",
        type="int",
        default=44, 
        help='t (s); '
    ) 
    group.add_option(
        "--dt",
        type="float",
        default=0.004,
        help='dt'
    )
    
    group.add_option(
        "--M",
        type="int",
        default=3,\
    )
    group.add_option(
        "--N",
        default=[400, 400, 200], 
    )
    group.add_option(
        "--Nsampled",
        type=int,
        default=3, 
    )
    group.add_option(
        "--B",
        type=int,
        default=1, 
    )

    group.add_option(
        "--asyn",
        default=[1000/3., 1000/3., 500/3.], 
    ) 
    group.add_option(
        "--eps",
        type="float",
        default=1., 
    ) 
    group.add_option(
        "--ref_t",
        type="float",
        default=0.004, 
    )  
    group.add_option(
        "--syn_delay",
        type="float",
        default=0.0, 
    )  
    group.add_option(
        "--conmat",
        default=[[1, 0., 1], [0., 1, 1], [-1, -1, -1]], 
    )  
    group.add_option(
        "--a_cutoff",
        type="float",
        default=1.0,
        help='age cutoff (s)'
    )
    group.add_option(
        "--initialize",
        type="str",
        default='stationary', 
    ) 
    
    parser.add_option_group(group)



    group = OptionGroup(parser, "ground_truth")
    group.add_option(
        "--amem",
        default=[50]*3, 
    ) 
    group.add_option(
        "--J",
        default=[9.984, 0, 9.984,0,9.984, 9.984,19.968, 19.968, 19.968, ], # ex2ex, ex2inh, inh2ex, inh2inh
        # default=[9.984, 9.984, 19.968, ], # ex2ex, ex2inh, inh2ex, inh2inh
    ) 
    group.add_option(
        "--rp",
        default=[14.4]*3, 
    ) 
    group.add_option(
        "--ft",
        default=[3.7]*3, 
    )   
    parser.add_option_group(group)

    group = OptionGroup(parser, "initialization")
    group.add_option(
        "--alpha",
        type=float,
        default=0.4,  
    )
    group.add_option(
        "--beta",
        type=float,
        default=2.0,  
    )
    group.add_option(
        "--init_amem",
        default=[45]*3,  
        help='place holder, will be randomly sampled'
    ) 
    group.add_option(
        "--init_J",
        default=[15.]*3, 
        help='place holder, will be randomly sampled'
    ) 
    group.add_option(
        "--J_parameterize",
        type=int,
        default=3,  
        help='3 or 9',
    )
    group.add_option(
        "--init_rp",
        default=[20.]*3,  
        help='place holder, will be randomly sampled'
    ) 
    group.add_option(
        "--init_ft",
        default=[10.]*3,  
        help='place holder, will be randomly sampled'
    )   
    group.add_option(
        "--lambda_t",
        default=[1]*3, 
    )  
    group.add_option(
        "--sampled_spike_history_smoothed_w",
        type='int',
        default=400, 
        help='t (ms)', 
    )  
    group.add_option(
        "--sampled_spike_history_smoothed_kernel",
        default='gaussian',
    ) 
    group.add_option(
        "--train_folderbase",
        default='train',
    )  
    group.add_option(
        "--train_Nrinit",
        type='int',
        default=1,
    )  
    parser.add_option_group(group)

    group = OptionGroup(parser, "training")
    group.add_option(
        "--E_LR_optimal",
        type=float,
        default=1e-3, 
    )
    group.add_option(
        "--E_patience_optimal",
        type=int,
        default=3, 
    )
    group.add_option(
        "--epochE",
        type=int,
        default=200, 
    )
    group.add_option(
        "--epochM",
        type=int,
        default=200,  
    )
    group.add_option(
        "--max_epochs",
        type=int,
        default=20, 
        help='max number of EM iterations'
    )
    group.add_option(
        "--log_mode",
        default=0, 
        help='0 (py+pZ); 1(py only); 2(pZ only)',
    )
    group.add_option(
        "--inference_mode",
        default=False, 
    )
    group.add_option(
        "--update_popact",
        default=True, 
    )
    group.add_option(
        "--update_sampledneuron",
        default=True,
    )
    group.add_option(
        "--method",
        default='L-BFGS-B',
        help='method for M step minimize' 
    )
    group.add_option(
        "--SAVE_DIR",
        type="str",
        default='result/3pop example/', 
    ) 
    parser.add_option_group(group)

     
    group = OptionGroup(parser, "other settings")
    group.add_option(
        "--plot_act_smoothw",
        type="int",
        default=4, #ms
    ) 
    group.add_option(
        "--LOG_PER",
        type="int",
        default=1, 
    )
    parser.add_option_group(group)


    return parser



opt = parse_args()
T_perB = int(opt['t_perbatch']/opt['dt'])
T_total = int(opt['trial_length']/opt['dt'])
A = int(opt['a_cutoff']/opt['dt'])
B = opt['B']
Btotal = int(opt['trial_length']/opt['t_perbatch'])
assert B == Btotal



if opt['J_parameterize'] == 3:
    opt['J'] = [9.984,9.984,19.968]
    opt['init_J'] = [15.]*3
elif opt['J_parameterize'] == 9:
    opt['J'] = [9.984, 0, 9.984,0,9.984, 9.984,19.968, 19.968, 19.968, ]
    opt['init_J'] = [15.]*9
else:
    assert False

SAVE_DIR = f"{opt['SAVE_DIR']}data{opt['end_timepoint']}/{opt['train_folderbase']}{opt['train_Nrinit']}"
print(SAVE_DIR)


# used by all scripts
