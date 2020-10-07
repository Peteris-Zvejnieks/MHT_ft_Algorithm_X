import matplotlib.pyplot as plt
import numpy as np

from stat_funcs import movement_likelihood_func, new_or_gone_likelihood_func_Y, multi_bubble_likelihood_func
from associator import Associator, asc_condition, comb_constr
from optimizer import optimizer
from trajectories import node_trajectory_with_stats
from tracer import Tracer

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
I = 17

J = 0

Main_dir = main_dirs[I]
sub_dirs = glob.glob(Main_dir + '/*')
try: sub_dirs.remove(*glob.glob(Main_dir + '/**.ini'))
except: pass
Sub_dir  = sub_dirs[J]
print(Sub_dir)
#%%
Sig_displacement1 = 100
K1       = 0.6
Move   = movement_likelihood_func(Sig_displacement1, K1)

A = 0.1
Boundary= 35
Height  = 918
New    = new_or_gone_likelihood_func_Y(A, Boundary, 1)
Gone   = new_or_gone_likelihood_func_Y(-A, Height - Boundary, 0)

Sig_displacement2 = 60
K2 = 0.5

Merge  = multi_bubble_likelihood_func(Sig_displacement2, K2, 0)
Split  = multi_bubble_likelihood_func(Sig_displacement2, K2, 1)

Optimizer     = optimizer([Move, New, Gone, Merge, Split])
#%%
Max_displacement_per_frame = 300
Radius_multiplyer = 3
Min_displacement  = 30
Asc_condition  = asc_condition(Max_displacement_per_frame, Radius_multiplyer, Min_displacement)

Upsilon = 1.5
Mu_v = 100000000
Max_acc = 100
Comb_constr = comb_constr(Upsilon, Mu_v, Max_acc)

ASSociator = Associator(Asc_condition, Comb_constr, max_k = 1)
#%%
mu_V       = 60
sig_V      = 30
r_sig_S    = 0.4
node_trajectory = node_trajectory_with_stats(mu_V, sig_V, r_sig_S)

#%%
Max_occlusion = 3
Quantile = 0.1
tracer = Tracer(ASSociator, Optimizer, node_trajectory, Max_occlusion, Quantile, Sub_dir)
#%%
Indx = 6
Prepend = 'test_%i_'%Indx
tracer.dump_data('/'+Prepend+str(Max_occlusion), 15, 1)