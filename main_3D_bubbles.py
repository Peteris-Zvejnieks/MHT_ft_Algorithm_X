import matplotlib.pyplot as plt
import numpy as np

from stat_funcs import movement_likelihood_func, new_or_gone_likelihood_func, multi_bubble_likelihood_func
from associator import Associator as aAssociator
from associator import asc_condition_3D_bubbles, comb_constr
from optimizer import optimizer
from trajectories import node_trajectory_with_stats
from tracer import Tracer as tRacer

import glob
import os

plt.rcParams['figure.dpi'] = 300
np.set_printoptions(suppress=True)
#%%
drive = 'C:\\'
w_dir = drive + os.path.join(*(os.getcwd().split('\\')[1:-1] + ['Objects']))
#W_dir = r'C:\Users\FMOF\Documents\Work\Work Drive\Objects'
os.chdir(w_dir)
main_dirs = sorted(glob.glob('./*'))
#%%
I = 1

J = 0

sub_dirs = glob.glob(main_dirs[I] + '/*')
try: sub_dirs.remove(*glob.glob(main_dirs[I] + '/**.ini'))
except: pass
sub_dir = sub_dirs[J]
print(sub_dir)
del(I, J)
#%%
Sig_displacement1   = 0.01  #@param {type: "slider", min: 10, max: 100}
K1                  = 1 #@param {type:"slider", min:0, max:1, step:0.01}
move   = movement_likelihood_func(Sig_displacement1, K1)

A                   = 50 #@param {type:"slider", min:0.01, max:0.5, step:0.01}
Boundary            = 0.025 #@param {type:"slider", min:0, max:50}
Height              = 0.15 #@param {type:"slider", min:0, max:1500}
new    = new_or_gone_likelihood_func(A, Boundary, 1, 2)
gone   = new_or_gone_likelihood_func(-A, Height - Boundary, 0, 2)

Sig_displacement2   = 0.01 #@param {type:"slider", min:0, max:150}
K2                  = 0.5 #@param {type:"slider", min:0, max:1, step:0.01}
merge  = multi_bubble_likelihood_func(Sig_displacement2, K2, 0, 1)
split  = multi_bubble_likelihood_func(Sig_displacement2, K2, 1, 1)

oOptimizer     = optimizer([move, new, gone, merge, split])
#%%
Max_displacement_per_frame  = 0.008  #@param {type: "slider", min: 10, max: 500}
Radius_multiplyer           = 4 #@param {type:"slider", min:1, max:10}
Min_displacement            = 0.001 #@param {type:"slider", min:0, max:100}
assc_condition  = asc_condition_3D_bubbles(Max_displacement_per_frame, Radius_multiplyer, Min_displacement)

Upsilon                     = 4 #@param {type:"slider", min:0.01, max:1.5, step:0.01}
K_v                         = 10 #@param {type:"slider", min:0, max:300}
Max_acc                     = 5 #@param {type:"slider", min:0, max:300}
cComb_constr = comb_constr(Upsilon, K_v, Max_acc)

aSSociator = aAssociator(assc_condition, cComb_constr, max_k=1)
#%%
Mu_V       = 0.0025 #@param {type:"slider", min:0, max:100}
Sig_V      = 0.01 #@param {type:"slider", min:0, max:100}
R_sig_S    = 0.5 #@param {type:"slider", min:0.01, max:1.5, step:0.01}
node_trajectory = node_trajectory_with_stats(Mu_V, Sig_V, R_sig_S)
#%%
Max_occlusion = 1
Quantile = 0.05
#%%
tracer = tRacer(aSSociator, oOptimizer, node_trajectory, Max_occlusion, Quantile, sub_dir,3)
#%%
indx = 10
string = '/'+'test_%i_'%indx+str(Max_occlusion)
tracer.dump_data(string, 15, 1)
#%%
parameters = {name: eval(name) for name in dir() if name[0].isupper() and name != 'In' and name != 'Out'}
import json
with open(sub_dir + '/Tracer Output' + '/'+string + '/parameters.json', 'w') as fp: json.dump(parameters, fp)