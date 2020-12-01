import matplotlib.pyplot as plt
import numpy as np

from stat_funcs import new_or_gone_likelihood_func, movement_likelihood_func
from associator import Associator as aAssociator
from associator import asc_condition_particles, comb_constr
from optimizer import optimizer
from trajectories import node_trajectory_with_stats
from tracer import Tracer as tTracer

import glob
import os

plt.rcParams['figure.dpi'] = 500
np.set_printoptions(suppress=True)
#%%
drive = 'C:\\'
w_dir = drive + os.path.join(*(os.getcwd().split('\\')[1:-1] + ['Objects']))
os.chdir(w_dir)
main_dirs = sorted(glob.glob('./*'))
#%%
I = 12

J = 0

sub_dirs = glob.glob(main_dirs[I] + '/*')
try: sub_dirs.remove(*glob.glob(main_dirs[I] + '/**.ini'))
except: pass
sub_dir = sub_dirs[J]
print(sub_dir)
del(I, J)
#%%
Sig_displacement1   = 8
K1                  = 0.4
move    = movement_likelihood_func(Sig_displacement1, K1)

A                   = 0.04

Boundary            = 15
width              = 480
new     = new_or_gone_likelihood_func(A, Boundary, 1, 0)
gone    = new_or_gone_likelihood_func(-A, width - Boundary, 0, 0)

oOptimizer   = optimizer([move, new, gone])
#%%
Max_displ_per_frame = 30
Radius_multlplyer  = 3
Min_displacement = 3
assc_condition  = asc_condition_particles(Max_displ_per_frame, Radius_multlplyer, Min_displacement)

Upsilon = 3
Mu_v = 30
Max_acc = 5
cComb_constr = comb_constr(Upsilon, Mu_v, Max_acc)

aSSociator = aAssociator(assc_condition, cComb_constr, max_k = 1)
#%%
K_V       = 100 #@param {type:"slider", min:0, max:100}
Sig_V      = 6 #@param {type:"slider", min:0, max:100}
R_sig_S    = 1 #@param {type:"slider", min:0.01, max:1.5, step:0.01}
trajectory_stats = node_trajectory_with_stats(K_V, Sig_V, R_sig_S)
#%%
Max_occlusion = 3
Quantile = 0.5
#%%
tracer = tTracer(aSSociator, oOptimizer, trajectory_stats,
                Max_occlusion, Quantile, sub_dir)
#%%
indx = 3
string = '/'+'test_new_constr _%i_'%indx+str(Max_occlusion)
#%%
# tracer.dump_data(string, 15, 5)
# #%%
# parameters = {name: eval(name) for name in dir() if name[0].isupper() and name != 'In' and name != 'Out'}
# import json
# with open(sub_dir + '/Tracer Output' + '/'+string + '/parameters.json', 'w') as fp: json.dump(parameters, fp)
