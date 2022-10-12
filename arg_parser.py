from optparse import OptionParser
from optparse import OptionGroup
import time


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
        default=11, 
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
        help='1/tau_mem'
    ) 
    group.add_option(
        "--J",
        default=[9.984,9.984,19.968]
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
        "--lb",
        type=float,
        default=0.4,  
        help='parameters are assumed to follow the uniform prior of [lb, ub]*ground_truth'
    )
    group.add_option(
        "--ub",
        type=float,
        default=2.0,  
        help='parameters are assumed to follow the uniform prior of [lb, ub]*ground_truth'
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
        "--train_Noinit",
        type='int',
        default=0,
    )  
    parser.add_option_group(group)

    group = OptionGroup(parser, "training")
    group.add_option(
        "--E_LR",
        type=float,
        default=1e-3, 
    )
    group.add_option(
        "--E_itertol",
        type=int,
        default=3, 
    )
    group.add_option(
        "--Emaxiters",
        type=int,
        default=200, 
    )
    group.add_option(
        "--Mmaxiters",
        type=int,
        default=200,  
    )
    group.add_option(
        "--max_EMsteps",
        type=int,
        default=20, 
        help='max number of EM iterations'
    )
    group.add_option(
        "--simulation_mode",
        default=False, 
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
        default=10, 
    )
    parser.add_option_group(group)


    return parser



opt = parse_args()

SAVE_DIR = f"{opt['SAVE_DIR']}data{opt['end_timepoint']}/{opt['train_folderbase']}{opt['train_Noinit']}"

T_perB = int(opt['t_perbatch']/opt['dt'])
A = int(opt['a_cutoff']/opt['dt'])
B = opt['B']
assert B == int(opt['trial_length']/opt['t_perbatch'])

# used by all scripts
