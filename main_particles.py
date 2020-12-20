import matplotlib.pyplot as plt
import numpy as np

from stat_funcs import new_or_gone_likelihood_func, paricle_movement_likelihood
from associator import Associator as aAssociator
from associator import asc_condition_particles, comb_constr_particles
from optimizer import optimizer
from trajectories import particle_trajectory_with_default_stats
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
I = 1

J = 0

sub_dirs = glob.glob(main_dirs[I] + '/*')
try: sub_dirs.remove(*glob.glob(main_dirs[I] + '/**.ini'))
except: pass
sub_dir = sub_dirs[J]
print(sub_dir)
del(I, J)
#%%
Sig_displacement1   = 6
Sig_acc = 8
KK_v = 6
W1 = 0.6
W2 = 0.5
move    = paricle_movement_likelihood(Sig_displacement1, Sig_acc, KK_v, W1, W2)

A                   = 0.04

Boundary            = 15
width              = 480
new     = new_or_gone_likelihood_func(A, Boundary, 1, 0)
gone    = new_or_gone_likelihood_func(-A, width - Boundary, 0, 0)

oOptimizer   = optimizer([move, new, gone])
#%%
Soi = 17
assc_condition  = asc_condition_particles(Soi)

Mu_v = 22
Max_acc = 8
cComb_constr = comb_constr_particles(Mu_v, Max_acc)

aSSociator = aAssociator(assc_condition, cComb_constr, max_k = 1)
#%%
K_V       = 10 #@param {type:"slider", min:0, max:100}
Sig_V      = 8 #@param {type:"slider", min:0, max:100}
trajectory_stats = particle_trajectory_with_default_stats(K_V, Sig_V)
#%%
Max_occlusion = 2
Quantile = 0.6
#%%
tracer = tTracer(aSSociator, oOptimizer, trajectory_stats,
                Max_occlusion, Quantile, sub_dir)
#%%
indx = 20
string = '/'+'test_new_constr _%i_'%indx+str(Max_occlusion)
#%%
tracer.dump_data(string, 15, 10)
#%%
parameters = {name: eval(name) for name in dir() if name[0].isupper() and name != 'In' and name != 'Out'}
import json
with open(sub_dir + '/Tracer Output' + '/'+string + '/parameters.json', 'w') as fp: json.dump(parameters, fp)
