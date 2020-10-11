import matplotlib.pyplot as plt
import numpy as np

from stat_funcs import new_or_gone_likelihood_func_X, movement_likelihood_func
from associator import Associator, asc_condition_particles, comb_constr
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
I = 10
J = 0

Main_dir = main_dirs[I]
sub_dirs = glob.glob(Main_dir + '/*')
try: sub_dirs.remove(*glob.glob(Main_dir + '/**.ini'))
except: pass
Sub_dir  = sub_dirs[J]
print(Sub_dir)
#%%
Sig_displacement1   = 8
K1                  = 0.4
Move    = movement_likelihood_func(Sig_displacement1, K1)

A                   = 0.04

Boundary            = 15
width              = 480
New     = new_or_gone_likelihood_func(A, Boundary, 1)
Gone    = new_or_gone_likelihood_func(-A, width - Boundary, 0)

#   = 30
#K2                  = 0.7

#Merge   = multi_bubble_likelihood_func(Sig_displacement2, K2, 0)
#Split   = multi_bubble_likelihood_func(Sig_displacement2, K2, 1)

Stat_funcs  = [Move, New, Gone]#, Merge, Split]
Optimizer   = optimizer(Stat_funcs)
#%%
Max_displ_per_frame = 30
Radius_multlplyer  = 3
Min_displacement = 3
Asc_condition  = asc_condition_particles(Max_displ_per_frame, Radius_multlplyer, Min_displacement)

Upsilon = 3
Mu_v = 10
Max_acc = 5

Comb_constr = comb_constr(Upsilon, Mu_v, Max_acc)

ASSociator = Associator(Asc_condition, Comb_constr, max_k = 1)

Trajectory_stats = trajectory_stats_v1(10, 6, 0.5)
#%%
Max_occlusion = 3
Quantile = 0.5
tracer = Tracer(ASSociator, Optimizer, Trajectory_stats,
                Max_occlusion, Quantile, Sub_dir)
Indx = 29
Prepend = 'test_new_constr _%i_'%Indx
tracer.dump_data('/'+Prepend+str(Max_occlusion), 15, 5)
# import networkx as nx
# graph = tracer.graph
# subgraphs =