from visualizerV2 import Visualizer, Graph_interpreter
from PIL import Image
from tqdm import tqdm
import networkx as nx
import pandas as pd
import numpy as np
import imageio
import zipfile
import glob
import os
import io

class Tracer():
    def __init__(self,
                 associator,
                 optimizer,
                 node_trajectory,
                 max_occlusion,
                 quantile,
                 path):

        self.dataset            = np.array(pd.read_excel('%s\\dataset.xlsx'%path))
        self.path               = path
        self.optimizer          = optimizer
        self.associator         = associator
        self.node_trajectory    = node_trajectory

        self.time_range = np.array([np.min(self.dataset[:,0]), np.max(self.dataset[:,0])], dtype = int)
        self._initialize_graph()

        iterr = iter(Fib(max_occlusion))
        for i, window_width in enumerate(iterr):
            self._eradicate_unlikely_connections(quantile)
            self._main_loop(window_width)
            interpretation = Graph_interpreter(self.graph.copy(), self.special_nodes, self.node_trajectory)
            print('Trajectory count :' + str(len(interpretation.trajectories)))

    def _main_loop(self, window_width):
        for time in tqdm(range(self.time_range[0], self.time_range[1]), desc = 'Window width - %i'%window_width):
            group1, group2 = self._get_groups(time, time + window_width)
            if len(group1) == len(group2) == 0: continue

            ascs_4_opti, Ys, ascs = self.associator(group1, group2)
            ascs1, ascs2 = zip(*ascs)
            X, all_likelihoods, Likelihood = self.optimizer.optimize((ascs_4_opti, Ys, len(group1) + len(group2)))

            for parents, children, x, likelihood in zip(ascs1, ascs2, X, all_likelihoods):
                if not x: continue
                edges = []
                for parent in parents:
                    try: parent = group1[parent].nodes[-1]
                    except: p1, t1 = [0,0], -1e48
                    else: p1, t1 = self.data[parent][2:4], self.data[parent][0]
                    for child in children:
                        try: child = group2[child].nodes[0]
                        except: p2, t2 = [0,0], 1e48
                        else: p2, t2 = self.data[child][2:4], self.data[child][0]
                        v = (((p2[0] - p1[0])**2 + (p2[1] - p1[1])**2)**0.5)/(t2-t1)
                        edges.append((parent, child, {'likelihood': likelihood, 'velocity' : v}))

                if any([any([self.data[x[0]][0] == time     for x in edges]),
                        any([self.data[x[1]][0] == time + 1 for x in edges]),
                        likelihood > self.decision_boundary]):
                    self.graph.add_edges_from(edges)

    # def _get_groups(self, start, stop):
    #     nodes1, nodes2 = [], []
    #     for node in self.graph.nodes():
    #         if node in self.special_nodes: continue
    #         if list(self.graph.out_edges(node)) == [] and start <= self.data[node][0] <  stop: nodes1.append(node)
    #         if list(self.graph.in_edges(node))  == [] and start <  self.data[node][0] <= stop: nodes2.append(node)

    #     return(list(map(self._get_trajectory, nodes1)), list(map(self._get_trajectory, nodes2)))

    def _get_groups(self, start, stop):
        nodes1, nodes2 = [], []
        for node in self.graph.nodes():
            if node in self.special_nodes: continue
            if list(self.graph._succ[node]) == [] and start <= self.data[node][0] <  stop: nodes1.append(node)
            if list(self.graph._pred[node]) == [] and start <  self.data[node][0] <= stop: nodes2.append(node)
        return(list(map(self._get_trajectory, nodes1)), list(map(self._get_trajectory, nodes2)))

    def _get_trajectory(self, node0):
        nodes = [node0]
        functions = [lambda x: list(x[0] for x in self.graph.in_edges(x) if type(x[0]) is not str),
                     lambda x: list(x[1] for x in self.graph.out_edges(x) if type(x[1]) is not str)]
        direction = int(len(functions[0](node0)) > 0) - int(len(functions[1](node0)) > 0)

        if direction:
            i = int(direction > 0)
            f1, f2 = functions[1 - i], functions[i]
            while True:
                nodes_o_i = f1(nodes[-1])
                if len(nodes_o_i) == 1 and len(f2(nodes_o_i[0])) == 1:
                    nodes.append(nodes_o_i[0])
                else: break

        nodes.sort(key = lambda x: x[0])
        return self.node_trajectory(self.graph.subgraph(set(nodes)))

    def _eradicate_unlikely_connections(self, quantile):
        likelihoods             = nx.get_edge_attributes(self.graph, 'likelihood')
        self.decision_boundary  = np.quantile(np.array(list(likelihoods.values())), quantile)
        removables              = [edge for edge in self.graph.edges if likelihoods[edge] <= self.decision_boundary]
        self.graph.remove_edges_from(removables)

    def _initialize_graph(self):
        self.graph         = nx.DiGraph()
        self.special_nodes = ['Entry', 'Exit']
        self.graph.add_nodes_from(self.special_nodes, data = 'mommy calls me speshal')

        for adress, point in zip(np.array(self.dataset)[:,:2], np.array(self.dataset)):
            node = tuple(map(int, adress))
            self.graph.add_node(node, data = list(point))

            for i, special_node in enumerate(self.special_nodes):
                edge = [special_node, node]
                if i: edge.reverse()
                self.graph.add_edge(*tuple(edge), likelihood = float(point[0] == self.time_range[i]), velocity = 1e-12)

        self.data = nx.get_node_attributes(self.graph, 'data')

    def dump_data(self, sub_folder = None, memory = 15, smallest_trajectories = 1):
        self.images = unzip_images('%s\\Compressed Data\\Shapes.zip'%self.path)
        self.shape = self.images[0].shape

        if sub_folder is None:  output_path = self.path + '/Tracer Output'
        else:                   output_path = self.path + '/Tracer Output' + sub_folder
        try: os.makedirs(output_path)
        except: pass

        def save_func(path, imgs):
            try: os.mkdir(path)
            except FileExistsError:
                for old_img in glob.glob(path+'/**.jpg'): os.remove(old_img)
            for i, x in tqdm(enumerate(imgs), desc = 'Saving: ' + path.split('/')[-1]): imageio.imwrite(path+'/%i.jpg'%i, x)

        nx.readwrite.gml.write_gml(self.graph, output_path + '/graph.gml', stringizer = lambda x: str(x))
        interpretation = Graph_interpreter(self.graph, self.special_nodes, self.node_trajectory)
        Vis = Visualizer(self.images, interpretation)

        map(os.remove, glob.glob(output_path + '/trajectories/**.csv'))
        save_func(output_path + '/trajectories',      Vis.ShowTrajectories())
        for i, track in enumerate(interpretation.trajectories):
            with open(output_path + '/trajectories/data_%i.csv'%i, 'w'): pass
            np.savetxt(output_path + '/trajectories/data_%i.csv'%i, track.data, delimiter=",")

            with open(output_path + '/trajectories/changes_%i.csv'%i, 'w'): pass
            np.savetxt(output_path + '/trajectories/changes_%i.csv'%i, track.changes, delimiter=",")

        with open(output_path + '/trajectories/events.csv', 'w') as file:
            events_str = ''
            for event in interpretation.events: events_str += str(event) + '\n'
            file.write(events_str)

        save_func(output_path + '/families',          Vis.ShowFamilies('likelihood'))
        save_func(output_path + '/tracedIDs',         Vis.ShowHistory(memory, smallest_trajectories, 'ID'))
        save_func(output_path + '/traced_velocities', Vis.ShowHistory(memory, smallest_trajectories, 'velocity'))

        del Vis, self.images

def unzip_images(path):
    imgFromZip  = lambda name: np.repeat((np.asarray(Image.open(io.BytesIO(zp.read(name))), np.uint16))[:,:,np.newaxis],3,2)
    scaler      = lambda x: x * (2**-8 * int(np.max(x) >= 256) + int(np.max(x) < 256) + 254 * int(np.max(x) == 1))
    mapper      = lambda x: scaler(imgFromZip(x)).astype(np.uint8)
    with zipfile.ZipFile(path) as zp:
        names = zp.namelist()
        try:    names.remove(*[x for x in names if len(x.split('/')[-1]) == 0 or x.split(',')[-1] == 'ini'])
        except: pass
        names.sort(key = lambda x: int(x.split('/')[-1].split('_')[-1].split('.')[0]))
        images = list(map(mapper, tqdm(names, desc = 'Loading images ')))
    return images

class Fib:
    def __init__(self, maxx):
        self.max = maxx

    def __iter__(self):
        self.a = 1
        self.b = 1
        return self

    def __next__(self):
        fib = self.a
        if fib > self.max: raise StopIteration
        self.a, self.b = self.b, self.a + self.b
        return fib
