class rangee(list):
    def __init__(self, n):
        self.n = n
        lst = list(range(n))
        super().__init__([x for x in range(n)])

a = rangee(5)
print(a)
print(a.n)
#%%
class rng(list):
    def __init__(self, n):
        self.n = n
        super().__init__(range(n))

    def get(self, n):
        if n > self.n:
            print(self)
        else:
            for x in self:
                if x > n:
                    break
                print(x)
        print(self[1:5])

a = rng(10)
a.get(5)
a.get(11)
#%%

class subbb(list):
    def __init__(self, n):
        self.n = n
        super().__init__(range(n))

    def __sub__(self, other):
        return subbb(self.n - other.n)

b = subbb(5)
c =subbb(8)
c -= b
print(c, c.n)
#%%

class funcy(object):
    def __new__(cls, f, n):
        super(funcy, cls).__new__

    def __init__(cls, x):
        return cls.f(x*cls.n)


a = lambda x:x**2
f = funcy(a, 5)
print(f(5), f(3), f.n)

#%%
import glob
import os
dirr = 'C:/Users/FMOF/Documents/Work/Work Drive/Shapes/'
lst1 = glob.glob(dirr + '*')
lst2 = list(map(lambda x: dirr + x.split('\\')[-1].split(')')[-1], lst1))
#map(lambda x:os.rename(x[0], x[1]), zip(lst1, lst2))
for a, b in zip(lst1, lst2):
    os.rename(a, b)

#%%
if len(neighbors := list(range(5))) > 2:
       print(neighbors)
#%%
import  numpy as np
from matplotlib import cm
import matplotlib.pyplot as plt
import cv2

cmap = cm.plasma

shape = np.array([1260, 1000])
relative_size = np.array([0.15, 0.02])
cb_shape = (shape * relative_size).astype(np.int)

gradient = np.swapaxes(cmap(np.linspace(1, 0, cb_shape[0]))[:,:-1,np.newaxis],1,2)
colorbar = np.repeat(gradient, cb_shape[1], axis = 1)

relative_pos = np.array([0.8, 0.05])
pos = (shape * relative_pos).astype(np.int)

overlay = np.zeros(tuple(shape) + (3,))
overlay[pos[0]: pos[0] + cb_shape[0], pos[1]: pos[1] + cb_shape[1],:] = colorbar[:,:,:]

overlay = cv2.putText(overlay, '0',
                      (pos[1], pos[0] + cb_shape[0] + 25),
                      cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255))
overlay = cv2.putText(overlay, '1',
                      (pos[1], pos[0] - 10),
                      cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255))


plt.imshow(overlay)
#%%
lst = [1,2,3]
if (n := len(lst)) > 2:
    print(n*100)
print(n)
#%%

import numpy as np
class A(int):
    def __new__(cls):
        return super(A, cls).__new__()
    def __init__(self, n):
        super().__init__(n)

b= A(5)
#%%
import colorsys
import numpy as np
import matplotlib.pyplot as plt
class discrete_colormap():
    def __init__(self, N, /,
                 hue        = 1,
                 saturation = 0.8,
                 value      = 0.8):
        self.N   = N
        self.hue = hue
        self.sat = saturation
        self.val = value

    def _get_hue(self, n): return self.hue *  n / self.N
    def _get_sat(self):    return self.sat + (1 - self.sat) * np.random.random()
    def _get_val(self):    return self.val + (1 - self.val) * np.random.random()

    def __call__(self, n):
        return colorsys.hsv_to_rgb(self._get_hue(n), self._get_sat(), self._get_val())

N = 6000
dc = discrete_colormap(N)
height =  512*8
width = 4
shape = (height, width * N, 3)
CM = np.zeros(shape)
for n in range(N):
    CM[:, width * n:  width * n + width] = dc(n)
plt.imshow(CM)
#%%

from PIL import Image
import networkx as nx
import numpy as np
import imageio
import zipfile
import glob
import os
import io
diry = r'C:/Users/FMOF/Documents/Work/Work Drive/Objects/Particles/min_area=50'
diry = r'C:\Users\FMOF\Documents\Work\Work Drive\Objects\Area_Output - 2D Laminar Field OFF 2D_lm_animation_noField_120\Front View'
imgFromZip = lambda name: np.repeat(np.asarray(Image.open(io.BytesIO(zp.read(name))))[:,:,np.newaxis],3,2)
with zipfile.ZipFile('%s\\Compressed Data\\Shapes.zip'%diry) as zp:
    names = zp.namelist()[30:50]
    try: names.remove(*[x for x in names if len(x.split('/')[-1]) == 0])
    except: pass
    names.sort(key = lambda x: int(x.split('/')[-1].split('_')[-1].split('.')[0]))
    images = list(map(imgFromZip, names))