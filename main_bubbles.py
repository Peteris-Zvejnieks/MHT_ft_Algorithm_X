import matplotlib.pyplot as plt
import numpy as np

from stat_funcs import movement_likelihood_func, new_or_gone_likelihood_func, multi_bubble_likelihood_func
from associator import Associator, asc_condition, comb_constr
from optimizer import optimizer
from trajectories import node_trajectory_with_stats
from tracer import Tracer, unzip_images

import glob
import os

plt.rcParams['figure.dpi'] = 500
np.set_printoptions(suppress=True)
#%%
Drive = 'C:\\'
W_dir = Drive + os.path.join(*(os.getcwd().split('\\')[1:-1] + ['Objects']))
#W_dir = r'C:\Users\FMOF\Documents\Work\Work Drive\Objects'
os.chdir(W_dir)
main_dirs = sorted(glob.glob('./*'))
#%%
I = 6

J = 0

Main_dir = main_dirs[I]
sub_dirs = glob.glob(Main_dir + '/*')
try: sub_dirs.remove(*glob.glob(Main_dir + '/**.ini'))
except: pass
Sub_dir  = sub_dirs[J]
print(Sub_dir)
images = unzip_images('%s\\Compressed Data\\Shapes.zip'%Sub_dir)
#%%
Sig_displacement1   = 50 #@param {type: "slider", min: 10, max: 100}
K1                  = 0.1 #@param {type:"slider", min:0, max:1, step:0.01}
Move   = movement_likelihood_func(Sig_displacement1, K1)

A                   = 0.01 #@param {type:"slider", min:0.01, max:0.5, step:0.01}
Boundary            = 100 #@param {type:"slider", min:0, max:50}
Height              = 1806 #@param {type:"slider", min:0, max:1500}
New    = new_or_gone_likelihood_func(A, Boundary, 1)
Gone   = new_or_gone_likelihood_func(-A, Height - Boundary, 0)

power = 3/2
Sig_displacement2   = 50 #@param {type:"slider", min:0, max:150}
K2                  = 0.1 #@param {type:"slider", min:0, max:1, step:0.01}
Merge  = multi_bubble_likelihood_func(Sig_displacement2, K2, 0, power)
Split  = multi_bubble_likelihood_func(Sig_displacement2, K2, 1, power)

Optimizer     = optimizer([Move, New, Gone, Merge, Split])
#%%
Max_displacement_per_frame  = 300  #@param {type: "slider", min: 10, max: 500}
Radius_multiplyer           = 5 #@param {type:"slider", min:1, max:10}
Min_displacement            = 30 #@param {type:"slider", min:0, max:100}
Asc_condition  = asc_condition(Max_displacement_per_frame, Radius_multiplyer, Min_displacement)

Upsilon                     = 0.09 #@param {type:"slider", min:0.01, max:1.5, step:0.01}
Mu_v                        = 60 #@param {type:"slider", min:0, max:300}
Max_acc                     = 60 #@param {type:"slider", min:0, max:300}
Comb_constr = comb_constr(Upsilon, Mu_v, Max_acc)

ASSociator = Associator(Asc_condition, Comb_constr)
#%%
mu_V       = 50 #@param {type:"slider", min:0, max:100}
sig_V      = 30 #@param {type:"slider", min:0, max:100}
r_sig_S    = 25 #@param {type:"slider", min:0.01, max:1.5, step:0.01}
node_trajectory = node_trajectory_with_stats(mu_V, sig_V, r_sig_S)
#%%
Max_occlusion = 2
Quantile = 0.05
tracer = Tracer(ASSociator, Optimizer, node_trajectory, Max_occlusion, Quantile, Sub_dir)
#%%
Indx = 23
Prepend = 'test_%i_'%Indx
tracer.dump_data(images, '/'+Prepend+str(Max_occlusion), 15, 1)
