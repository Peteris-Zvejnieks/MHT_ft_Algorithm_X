import  numpy as np
from matplotlib import cm
from tqdm import tqdm
import colorsys
import copy
import cv2
import networkx as nx

class discrete_colormap():
    def __init__(self, N, /, hue = 1, sat = 0.8, val = 0.8):
        self.hue = lambda n: (hue *  n%N) /  N
        self.sat = lambda: sat + (1 - sat) * np.random.random()
        self.val = lambda: val + (1 - val) * np.random.random()
    def __call__(self, n): return colorsys.hsv_to_rgb(self.hue(n), self.sat(), self.val())

class Colorbar_overlay():
    def __init__(self, cmap, shape, /, relative_size = [0.15, 0.02], relative_pos = [0.6, 0.05]):
        relative_size, relative_pos, shapeXY  = np.array(relative_size), np.array(relative_pos), np.array(shape[:-1])
        self.cb_shape = (shapeXY * relative_size).astype(np.int)
        self.pos = (shapeXY * relative_pos).astype(np.int)

        gradient = np.swapaxes(cmap(np.linspace(1, 0, self.cb_shape[0]))[:,:-1,np.newaxis], 1, 2)
        colorbar = 255 * np.repeat(gradient, self.cb_shape[1], axis = 1)
        self.overlay = np.zeros(shape)
        self.overlay[self.pos[0]: self.pos[0] + self.cb_shape[0], self.pos[1]: self.pos[1] + self.cb_shape[1]] = colorbar

    def __call__(self, min_max):
        overlay = copy.deepcopy(self.overlay)
        overlay = cv2.putText(overlay, "{:.1e}".format(min_max[0]), (self.pos[1], self.pos[0] + self.cb_shape[0] + 30),    cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255))
        overlay = cv2.putText(overlay, "{:.1e}".format(min_max[1]), (self.pos[1], self.pos[0] - 10),                       cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255))
        return overlay

class Graph_interpreter():
    def __init__(self, graph, special_nodes, node_trajectory):
        self.graph = graph
        self.special_nodes = special_nodes
        self.node_trajectory = node_trajectory
        self._trajectories()

    def _find_by_node(self, node):
        for i, x in enumerate(self.trajectories):
            if node in x.nodes: return i

    def _trajectories(self):
        tmp_graph = self.graph.copy()
        tmp_graph.remove_nodes_from(self.special_nodes)
        for node in tmp_graph.nodes:
            if len(ins  := list(tmp_graph.in_edges( node))) > 1: tmp_graph.remove_edges_from(ins )
            if len(outs := list(tmp_graph.out_edges(node))) > 1: tmp_graph.remove_edges_from(outs)
        tmp_graph = tmp_graph.to_undirected()
        self.paths = list(self.graph.subgraph(c) for c in nx.connected_components(tmp_graph))
        self.paths.sort(key = lambda x: -len(x.nodes))
        self.trajectories = list(map(self.node_trajectory, self.paths))
        self.trajectories.sort(key = lambda x: x.data[0,0])

    def events(self):
        self.events = []
        tmp_graph = self.graph.copy()
        likelihoods = nx.get_edge_attributes(self.graph, 'likelihood')
        for entry in list(tmp_graph.out_edges(self.special_nodes[0])):   self.events.append([[self.special_nodes[0]],    [self._find_by_node(entry[1])],                 likelihoods[entry]])
        for exitt in list(tmp_graph.in_edges( self.special_nodes[1])):   self.events.append([[self._find_by_node(exitt[0])],                 [self.special_nodes[1]],    likelihoods[exitt]])
        tmp_graph.remove_nodes_from(self.special_nodes)
        for node in tmp_graph.nodes:
            if len(ins  := list(tmp_graph.in_edges( node))) > 1: self.events.append([list(map(lambda edge: self._find_by_node(edge[0]), ins)), [self._find_by_node(node)],  likelihoods[ ins[0]]])
            if len(outs := list(tmp_graph.out_edges(node))) > 1: self.events.append([[self._find_by_node(node)], list(map(lambda edge: self._find_by_node(edge[1]), outs)), likelihoods[outs[1]]])

    def families(self):
        tmp_graph = self.graph.copy().to_undirected()
        tmp_graph.remove_nodes_from(self.special_nodes)
        self.families = list(self.graph.subgraph(c.union(set(self.special_nodes))) for c in nx.connected_components(tmp_graph))
        self.families.sort(key = lambda graph: -len(list(graph.nodes)))

class Visualizer():
    def __init__(self, images, interpretation, /, width = 2):
        self.images = images
        self.shape  = images[0].shape
        self.interpretation = interpretation
        self.trajectories = self.interpretation.trajectories
        self.width = 2
        self.data = nx.get_node_attributes(self.interpretation.graph, 'data')

    def _get_node_crd(self, node):  return (int(self.data[node][2]), self.shape[0] - int(self.data[node][3]))
    def _map_color(self, color):    return tuple(map(lambda x: 255*x, color))[:3]
    def _normalizer(self, value, min_max):
        num = value - min_max[0]
        if (den := min_max[1] - min_max[0]) == 0:   return 1 - 1e-12
        else:                                       return min(num / den, 1 - 1e-12)

    def _draw_edge(self, img, edge, color):
        u, v = edge
        try: crd1 = self._get_node_crd(u)
        except: img = cv2.circle(img, self._get_node_crd(v), int(2 * self.width), color, self.width)
        else:
            try: crd2 = self._get_node_crd(v)
            except: img = cv2.circle(img, crd1, int(3 * self.width), color, self.width)
            else: img = cv2.line(img, crd1, crd2, color, self.width)
        finally: return img

    def ShowFamilies(self, key = 'likelihood'):
        cmap = cm.plasma
        color_bar_gen = Colorbar_overlay(cmap, self.shape)
        families = self.interpretation.families
        family_photos = np.zeros((len(families),) + self.shape, dtype = np.uint8)
        for i, family in tqdm(enumerate(families), desc = 'Drawing families '):
            values = nx.get_edge_attributes(family, key)
            min_max = [min(tuple(values.values()) + (1e3,)), max(tuple(values.values()) + (0,))]
            for edge in values.keys():
                color = self._map_color(cmap(self._normalizer(values[edge], min_max)))
                family_photos[i] = self._draw_edge(family_photos[i], edge, color)
            color_bar = color_bar_gen(min_max)
            family_photos[i] = np.where(color_bar != 0, color_bar, family_photos[i])
        return family_photos

    def ShowHistory(self, memory = 15, min_trajectory_size = 1, key = 'velocity'):
        events = self.interpretation.events
        images = copy.deepcopy(self.images)
        dcmap = discrete_colormap(memory * 2)
        cmap = cm.plasma
        if key != 'ID': values = nx.get_edge_attributes(self.interpretation.graph, key)
        for i, tr in tqdm(enumerate(self.trajectories), desc = 'Drawing ' + key + ' history '):
            if len(tr) < min_trajectory_size: continue
            if key != 'ID': min_max = [min(tuple(values.values()) + (1e3,)), max(tuple(values.values()) + (0,))]
            color = self._map_color(dcmap(i))
            for edge in zip(tr.nodes[:-1], tr.nodes[1:]):
                t0 = int(self.data[edge[1]][0])
                if key != 'ID': color = self._map_color(cmap(self._normalizer(values[edge], min_max)))
                for t in range(t0, min(len(images), t0 + memory)): images[t - 1] = self._draw_edge(images[t - 1], edge, color)
        for event in events:
            stops, starts, likelihood = event
            if type(stops[0]) is str:
                ID = starts[0]
                if len(self.trajectories[ID]) < min_trajectory_size: continue
                R, color = int(self.width * 2), (0,  0, 255)
                t0 = int(self.trajectories[starts[0]].data[0, 0])
                crd = self._get_node_crd(self.trajectories[ID].nodes[0])
                f = lambda x: cv2.circle(x, crd, R, color, self.width)
            elif type(starts[0]) is str:
                ID = stops[0]
                if len(self.trajectories[ID]) < min_trajectory_size: continue
                R, color = int(self.width * 3), (0, 255, 0)
                t0 = int(self.trajectories[ID].data[-1, 0])
                crd = self._get_node_crd(self.trajectories[ID].nodes[-1])
                f = lambda x: cv2.circle(x, crd, R, color, self.width)
            elif len(stops) > 1:
                t0 = int(self.trajectories[starts[0]].data[0, 0])
                f = lambda x: self._draw_merger(x, stops, starts[0])
            elif len(starts) > 1:
                t0 = int(self.trajectories[stops[0]].data[-1,0])
                f = lambda x: self._draw_split(x, stops[0], starts)
            for t in range(t0, min(len(images), t0 + memory)): images[t - 1] = f(images[t - 1])
        return images

    def _draw_merger(self, img, starts, stop):
        color=  (255, 255, 0)
        crd2 = self._get_node_crd(self.trajectories[stop].nodes[0])
        for start in starts:
            crd1 = self._get_node_crd(self.trajectories[start].nodes[-1])
            img = cv2.arrowedLine(img, crd1, crd2, color, self.width)
        return img

    def _draw_split(self, img, start, stops):
        color = (255, 0, 255)
        crd1 = self._get_node_crd(self.trajectories[start].nodes[-1])
        for stop in stops:
            crd2 = self._get_node_crd(self.trajectories[stop].nodes[0])
            img = cv2.arrowedLine(img, crd1, crd2, color, self.width)
        return img

    def ShowTrajectories(self, key = 'velocity'):
        cmap = cm.plasma
        color_bar_gen = Colorbar_overlay(cmap, self.shape)
        imgs = np.zeros((len(self.trajectories),) + self.shape, dtype=np.uint8)
        for i, tr in tqdm(enumerate(self.trajectories), desc = 'Drawing trajectories '):
            values = nx.get_edge_attributes(tr.backbone, key)
            min_max = [min(tuple(values.values()) + (1e3,)), max(tuple(values.values()) + (0,))]
            for edge in zip(tr.nodes[:-1], tr.nodes[1:]):
                color = self._map_color(cmap(self._normalizer(values[edge], min_max)))
                imgs[i] = self._draw_edge(imgs[i], edge, color)
            color_bar = color_bar_gen(min_max)
            imgs[i] = np.where(color_bar != 0, color_bar, imgs[i])
        return imgs
