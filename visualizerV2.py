import  numpy as np
from matplotlib import cm
import itertools as itt
from tqdm import tqdm
import colorsys
import copy
import cv2
import networkx as nx

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
        return colorsys.hsv_to_rgb(self._get_hue(n%self.N), self._get_sat(), self._get_val())

class Colorbar_overlay():
    def __init__(self, cmap, shape, /, relative_size = [0.15, 0.02], relative_pos = [0.6, 0.05]):
        relative_size, relative_pos  = np.array(relative_size), np.array(relative_pos)
        = 
        shape = np.array(shape)

        self.cb_shape = (shape * relative_size).astype(np.int)
        self.pos = (shape * relative_pos).astype(np.int)

        gradient = np.swapaxes(cmap(np.linspace(1, 0, self.cb_shape[0]))[:,:-1,np.newaxis], 1, 2)
        colorbar = 255 * np.repeat(gradient, self.cb_shape[1], axis = 1)
        self.overlay = np.zeros(shape)
        self.overlay[self.pos[0]: self.pos[0] + self.cb_shape[0], self.pos[1]: self.pos[1] + self.cb_shape[1],:] = colorbar[:,:,:]

    def __call__(self, min_max):
        overlay = cv2.putText(self.overlay, "{:.1e}".format(min_max[0]),
                              (self.pos[1], self.pos[0] + self.cb_shape[0] + 30),
                              cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255))

        overlay = cv2.putText(overlay, "{:.1e}".format(min_max[1]),
                              (self.pos[1], self.pos[0] - 10),
                              cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255))
        return overlay

class Visualizer():
    def __init__(self, images, graph):
        self.graph = graph
        self.images = images