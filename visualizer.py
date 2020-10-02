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

class visualizer():
    def __init__(self, images, trajectories):
        self.width  = 2
        self.images = images
        self.shape  = images[0].shape
        self.trajectories = trajectories

    def ShowTrajectories(self):
        imgs = np.zeros((len(self.trajectories),) + self.shape, dtype=np.uint8)
        for i, trajectory in tqdm(enumerate(self.trajectories), desc = 'Drawing trajectories '):
            imgs[i] = self._draw_trajectory(imgs[i], trajectory.data,
                                            trajectory.likelihoods,
                                            cm.plasma)
        return imgs

    def ShowHistory(self, memory = 15, smallest_trajectories = 1, coloring_scheme = 'velocity'):
        images = copy.deepcopy(self.images)
        dc = discrete_colormap(memory * 2)

        for i, trajectory in tqdm(enumerate(self.trajectories), desc = 'Drawing '+coloring_scheme+' history '):

            if len(trajectory) < smallest_trajectories: continue

            colors_vals = {'velocities' : np.linalg.norm(trajectory.changes[:,1:3], axis = 1)/trajectory.changes[:,0],
                           'likelihood': trajectory.likelihoods,
                           'ID'         : itt.repeat(i)}[coloring_scheme]

            min_max = [0, 1]
            if coloring_scheme == 'velocities'  or coloring_scheme == 'likelihood':
                 min_max = [min(tuple(colors_vals) + (1e3,)),
                            max(tuple(colors_vals) + (  0,))]

            cf = {'velocities' : lambda x: cm.plasma(min(self._normalizer(x, min_max), 1 - 1e-12)),
                  'likelihoods': lambda x: cm.plasma(min(self._normalizer(x, min_max), 1 - 1e-12)),
                  'ID'         : lambda i: dc(i)}[coloring_scheme]

            for p1, p2, val in zip(trajectory.data[:-1], trajectory.data[1:], colors_vals):
                for img_ID in range(int(p1[0]), min(len(images), int(p2[0] + memory))):
                    crd1 = self._get_crd(p1)
                    crd2 = self._get_crd(p2)
                    color = self._map_color(cf(val))
                    images[img_ID] = cv2.line(images[img_ID], crd1, crd2, color, self.width)

        for event in self.trajectories.events:
            stops, starts, likelihood = event

            if type(stops[0]) is str:
                ID = starts[0]
                if len(self.trajectories[ID]) < smallest_trajectories: continue
                R, color = int(self.width * 2), (0,  0, 255)
                frame_ID = int(self.trajectories[starts[0]].data[0, 0])
                crd = self._get_crd(self.trajectories[ID].data[0,:])
                f = lambda x: cv2.circle(x, crd, R, color, self.width)

            elif type(starts[0]) is str:
                ID = stops[0]
                if len(self.trajectories[ID]) < smallest_trajectories: continue
                R, color = int(self.width * 3), (0, 255, 0)
                frame_ID = int(self.trajectories[ID].data[-1, 0])
                crd = self._get_crd(self.trajectories[ID].data[-1,:])
                f = lambda x: cv2.circle(x, crd, R, color, self.width)

            elif len(stops) > 1:
                frame_ID = int(self.trajectories[starts[0]].data[0, 0])
                f = lambda x: self._draw_merger(x, stops, starts[0])

            elif len(starts) > 1:
                frame_ID = int(self.trajectories[stops[0]].data[-1,0])
                f = lambda x: self._draw_split(x, stops[0], starts)

            else: raise StopIteration('Dumb event: {}'.format(event))
            for img_ID in range(frame_ID-1, min(len(images), frame_ID-1 + memory)):
                images[img_ID] = f(images[img_ID])

        return images

    def ShowFamilies(self):
        cmap = cm.plasma
        families = self.trajectories.families
        self.family_photos = np.zeros((len(families),) + self.shape, dtype = np.uint8)

        for i, family in tqdm(enumerate(families), desc = 'Drawing families '):
            likelihoods = nx.get_edge_attributes(family, 'likelihood')
            data        = nx.get_node_attributes(family, 'data')

            min_max     = [min( tuple(likelihoods.values()) + (1,)),
                           max( tuple(likelihoods.values()) + (0,))]

            for edge in likelihoods.keys():
                norm_likelihood = self._normalizer(likelihoods[edge], min_max)

                u, v = edge
                color = self._map_color(cmap(min(norm_likelihood, 1 - 1e-12)))

                if type(u) is str:
                    if u != 'new': raise KeyError('fek1')
                    self.family_photos[i,:,:] = cv2.circle(self.family_photos[i,:,:],
                                                           self._get_crd(data[v]),
                                                           int(2 * self.width),
                                                           color, self.width)
                elif type(v) is str:
                    if v != 'gone': raise KeyError('fek2')
                    self.family_photos[i,:,:] = cv2.circle(self.family_photos[i,:,:],
                                                           self._get_crd(data[u]),
                                                           int(3 * self.width),
                                                           color, self.width)
                else:
                    p1 = data[u]
                    p2 = data[v]
                    self.family_photos[i,:,:] = cv2.line(self.family_photos[i,:,:],
                                                         self._get_crd(p1), self._get_crd(p2),
                                                         color, self.width)

            colorbar_overlay = self._colorbar(cmap, min_max)
            self.family_photos[i,:,:] = np.where(colorbar_overlay != 0, colorbar_overlay, self.family_photos[i,:,:])

        return self.family_photos

    def _colorbar(self, cmap, min_max):

        relative_size = np.array([0.15, 0.02])
        relative_pos = np.array([0.6, 0.05])
        shape = np.array(self.shape[:-1])

        cb_shape = (shape * relative_size).astype(np.int)
        pos = (shape * relative_pos).astype(np.int)

        gradient = np.swapaxes(cmap(np.linspace(1, 0, cb_shape[0]))[:,:-1,np.newaxis], 1, 2)
        colorbar = 255 * np.repeat(gradient, cb_shape[1], axis = 1)

        overlay = np.zeros(self.shape)
        overlay[pos[0]: pos[0] + cb_shape[0], pos[1]: pos[1] + cb_shape[1],:] = colorbar[:,:,:]

        overlay = cv2.putText(overlay, "{:.1e}".format(min_max[0]),
                              (pos[1], pos[0] + cb_shape[0] + 30),
                              cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255))
        overlay = cv2.putText(overlay, "{:.1e}".format(min_max[1]),
                              (pos[1], pos[0] - 10),
                              cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255))
        return overlay

    def _normalizer(self, likelihood, min_max):
        num = likelihood - min_max[0]
        den = min_max[1] - min_max[0]
        if den == 0: return 1
        else:        return num / den

    def _draw_trajectory(self, img, trajectory, likelihoods, colorMap):
        for p1, p2, likelihood in zip(trajectory[:-1], trajectory[1:], likelihoods):
            color = self._map_color(colorMap(float(likelihood)))
            img = cv2.line(img, self._get_crd(p1), self._get_crd(p2), color, self.width)
        return img

    def _draw_connection(self, img, p1, p2, likelihood, colorMap):
        color = self._map_color(colorMap(likelihood))
        crd1 = self._get_crd(p1)
        crd2 = self._get_crd(p2)
        return cv2.arrowedLine(img, crd1, crd2, color, self.width)

    def _draw_merger(self, img, starts, stop):
        color=  (255, 255, 0)
        crd2 = self._get_crd(self.trajectories[stop].data[0])
        for start in starts:
            crd1 = self._get_crd(self.trajectories[start].data[-1])
            img = cv2.arrowedLine(img, crd1, crd2, color, self.width)
        return img

    def _draw_split(self, img, start, stops):
        color = (255, 0, 255)
        crd1 = self._get_crd(self.trajectories[start].data[-1])
        for stop in stops:
            crd2 = self._get_crd(self.trajectories[stop].data[0])
            img = cv2.arrowedLine(img, crd1, crd2, color, self.width)
        return img

    def _get_crd(self, point):
        return (int(point[2]), self.shape[0] - int(point[3]))

    def _map_color(self, color):
        RGB = list(map(lambda x: 255*x, color))[:3]
        return tuple(RGB)
