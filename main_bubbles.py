import matplotlib.pyplot as plt
import numpy as np

from stat_funcs import movement_likelihood_func, new_or_gone_likelihood_func, multi_bubble_likelihood_func
from associator import Associator as aAssociator
from associator import asc_condition, comb_constr
from optimizer import optimizer
from trajectories import node_trajectory_with_stats
from tracer import unzip_images
from tracer import Tracer as tTracer

import glob
import os

plt.rcParams['figure.dpi'] = 500
np.set_printoptions(suppress=True)
#%%
drive = 'C:\\'
w_dir = drive + os.path.join(*(os.getcwd().split('\\')[1:-1] + ['Objects']))
#W_dir = r'C:\Users\FMOF\Documents\Work\Work Drive\Objects'
os.chdir(w_dir)
main_dirs = sorted(glob.glob('./*'))
#%%
I = 3

J = 0

sub_dirs = glob.glob(main_dirs[I] + '/*')
try: sub_dirs.remove(*glob.glob(main_dirs[I] + '/**.ini'))
except: pass
sub_dir  = sub_dirs[J]
print(sub_dir)
images = unzip_images('%s\\Compressed Data\\Shapes.zip'%sub_dir)
del(I, J)
#%%
Sig_displacement_translation   = 50 #@param {type: "slider", min: 10, max: 100}
K_translation                  = 0.1 #@param {type:"slider", min:0, max:1, step:0.01}
move   = movement_likelihood_func(Sig_displacement_translation, K_translation)

A                   = 0.01 #@param {type:"slider", min:0.01, max:0.5, step:0.01}
Boundary            = 100 #@param {type:"slider", min:0, max:50}
height              = 1806 #@param {type:"slider", min:0, max:1500}
new    = new_or_gone_likelihood_func(A, Boundary, 1)
gone   = new_or_gone_likelihood_func(-A, height - Boundary, 0)

Power = 3/2
Sig_displacement_split_merge  = 50 #@param {type:"slider", min:0, max:150}
K_split_merge                  = 0.1 #@param {type:"slider", min:0, max:1, step:0.01}
merge  = multi_bubble_likelihood_func(Sig_displacement_split_merge, K_split_merge, 0, Power)
split  = multi_bubble_likelihood_func(Sig_displacement_split_merge, K_split_merge, 1, Power)

oOptimizer     = optimizer([move, new, gone, merge, split])
#%%
Max_displacement_per_frame  = 400  #@param {type: "slider", min: 10, max: 500}
Radius_multiplyer           = 5 #@param {type:"slider", min:1, max:10}
Min_displacement            = 30 #@param {type:"slider", min:0, max:100}
aAsc_condition  = asc_condition(Max_displacement_per_frame, Radius_multiplyer, Min_displacement)

Upsilon                     = 0.09 #@param {type:"slider", min:0.01, max:1.5, step:0.01}
K_v                         = 60 #@param {type:"slider", min:0, max:300}
Max_acc                     = 60 #@param {type:"slider", min:0, max:300}
cComb_constr = comb_constr(Upsilon, K_v, Max_acc)

aASSociator = aAssociator(aAsc_condition, cComb_constr)
#%%
Mu_V       = 50 #@param {type:"slider", min:0, max:100}
Sig_V      = 30 #@param {type:"slider", min:0, max:100}
R_sig_S    = 25 #@param {type:"slider", min:0.01, max:1.5, step:0.01}
node_trajectory = node_trajectory_with_stats(Mu_V, Sig_V, R_sig_S)
#%%
Max_occlusion = 2
Quantile = 0.05
tracer = tTracer(aASSociator, oOptimizer, node_trajectory, Max_occlusion, Quantile, sub_dir)
#%%
indx = 1
prepend = 'test_%i_'%indx
tracer.dump_data(images, '/'+prepend+str(Max_occlusion), 15, 1)
#%%
parameters = {name: eval(name) for name in dir() if name[0].isupper() and name != 'In' and name != 'Out'}
import json
with open(sub_dir + '/Tracer Output' + '/'+prepend + str(Max_occlusion) + '/parameters.json', 'w') as fp: json.dump(parameters, fp)

