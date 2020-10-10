import  numpy as np
from matplotlib import cm
import itertools as itt
from tqdm import tqdm
import colorsys
import copy
import cv2
import networkx as nx

class discrete_colormap():
    def __init__(self, N, /, hue = 1, sat = 0.8, value = 0.8):
        self.sat = saturation
        self.val = value
        self._get_hue = lambda n: (hue *  n%N) /  N
        self._get_sat = lambda: sat + (1 - sat) * np.random.random()
        self._get_val = lambda: val + (1 - val) * np.random.random()

    def __call__(self, n):
        return colorsys.hsv_to_rgb(self._get_hue(n), self._get_sat(), self._get_val())

class Colorbar_overlay():
    def __init__(self, cmap, shape, /, relative_size = [0.15, 0.02], relative_pos = [0.6, 0.05]):
        relative_size, relative_pos  = np.array(relative_size), np.array(relative_pos)
        shape = np.array(shape)

        self.cb_shape = (shape * relative_size).astype(np.int)
        self.pos = (shape * relative_pos).astype(np.int)

        gradient = np.swapaxes(cmap(np.linspace(1, 0, self.cb_shape[0]))[:,:-1,np.newaxis], 1, 2)
        colorbar = 255 * np.repeat(gradient, self.cb_shape[1], axis = 1)
        self.overlay = np.zeros(shape)
        self.overlay[self.pos[0]: self.pos[0] + self.cb_shape[0], self.pos[1]: self.pos[1] + self.cb_shape[1],:] = colorbar[:,:,:]
        del(gradient, colorbar, relative_pos, relative_size, shape)

    def __call__(self, min_max):
        overlay = cv2.putText(self.overlay, "{:.1e}".format(min_max[0]),
                              (self.pos[1], self.pos[0] + self.cb_shape[0] + 30),
                              cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255))

        overlay = cv2.putText(overlay, "{:.1e}".format(min_max[1]),
                              (self.pos[1], self.pos[0] - 10),
                              cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255))
        return overlay

class Graph_interpreter():
    def __init__(self, graph, special_nodes, node_trajectory):
        self.graph = graph
        self.special_nodes = special_nodes
        self.node_trajectory = node_trajectory
        self._trajectories()
        self._events()
        self._families()

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
        self.trajectories = map(self.node_trajectory, self.paths)
        del(tmp_graph, c, ins, outs)

    def _events(self):
        self.events = []
        likelihoods = nx.get_edge_attributes(self.digraph, 'likelihood')
        for entry in list(self.graph.out_edges(self.special_nodes[0])):   self.events.append([[self.special_nodes[0]],    [entry[1]],                 likelihoods[entry]])
        for exit  in list(self.graph.in_edges( self.special_nodes[1])):   self.events.append([[exit[0]],                  [self.special_nodes[1]],    likelihoods[exit]])
        self.graph.remove_nodes_from(self.special_nodes)
        for node in self.graph.nodes: 
            if len(ins  := list(self.graph.in_edges( node))) > 1: self.events.append([list(map(lambda edge: self._find_by_node(edge[0]), ins)), [self._find_by_node(node)],  likelihoods[ ins[0]]])
            if len(outs := list(self.graph.out_edges(node))) > 1: self.events.append([[self._find_by_node(node)], list(map(lambda edge: self._find_by_node(edge[1]), outs)), likelihoods[outs[1]]])
        del(entry, exit, ins, outs, node, likelihoods)

    def _families(self):
        tmp_graph = self.graph.copy().to_undirected()        
        tmp_graph.remove_nodes_from(self.special_nodes)        
        self.families = list(self.graph.subgraph(c.union(set(self.special_nodes))) for c in nx.connected_components(tmp_graph))
        self.families.sort(key = lambda graph: -len(list(graph.nodes)))
        del(tmp_graph, c)

class Visualizer():
    def __init__(self, images, interpretation, /, width = 2):
        self.images = images
        self.shape  = images[0].shape
        self.interpretation = interpretation
        self.widht = 2
        self.data = nx.get_node_atributes(self.interpretation.graph, 'data')

    def _get_node_crd(self, node): return (int(self.data[node][2]), self.shape[0] - int(self.data[node][3]))

    def _draw_edge(self, img, edge, color):
        u, v = edge
        try: crd1 = self._get_node_crd(u)
        except: img = cv2.circle(img, self._get_node_crd(v), int(2 * self.width), color, self.width)
        try: crd2 = self._get_node_crd(v)
        except: img = cv2.circle(img, crd1, int(3 * self.width), color, self.width)


    def _map_color(self, color): return tuple(map(lambda x: 255*x, color))[:3]

    def _normalizer(self, value, min_max):
        num = value - min_max[0]
        if (den := min_max[1] - min_max[0]) == 0:   return 1
        else:                                       return num / den

    def ShowFamilies(self):
        cmap = cm.plasma
        families = self.interpretation.families
        self.family_photos = np.zeros((len(families),) + self.shape, dtype = np.uint8)
        for i, family in tqdm(enumerate(families), desc = 'Drawing families '):
            likelihoods = nx.get_edge_attributes(family, 'likelihood')
            min_max     = [min( tuple(likelihoods.values()) + (1,)), max( tuple(likelihoods.values()) + (0,))]
            for edge in likelihoods.keys():
                color = self._map_color(min(1 - 1e-12, self._normalizer(likelihoods[edge], min_max)))



