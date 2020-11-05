from visualizerV2 import Visualizer, Graph_interpreter
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
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
                 path,
                 dim = 2):


        dataset                 = pd.read_csv('%s\\dataset.csv'%path)
        self.columns            = dataset.columns
        self.dataset            = np.array(dataset)
        self.path               = path
        self.optimizer          = optimizer
        self.associator         = associator
        self.node_trajectory    = node_trajectory
        self.dim                = dim

        self.time_range = np.array([np.min(self.dataset[:,0]), np.max(self.dataset[:,0])], dtype = int)

        self._initialize_graph()
        self._window_sweep(max_occlusion, quantile)

    def _window_sweep(self, max_occlusion, quantile):
        iterr = iter(Fib(max_occlusion))
        for i, window_width in enumerate(iterr):
            self._eradicate_unlikely_connections(quantile)
            self._time_sweep(window_width)
            interpretation = Graph_interpreter(self.graph.copy(), self.special_nodes, self.node_trajectory)
            print('Trajectory count :' + str(len(interpretation.trajectories)))

    def _time_sweep(self, window_width):
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
                    except: p1, t1 = np.zeros(2), -1e48
                    else: p1, t1 = np.array(self.data[parent][2:4]), self.data[parent][0]
                    for child in children:
                        try: child = group2[child].nodes[0]
                        except: p2, t2 = np.zeros(2), 1e48
                        else: p2, t2 = np.array(self.data[child][2:4]), self.data[child][0]
                        v = np.dot(p1, p2) / (t2-t1)
                        edges.append((parent, child, {'likelihood': likelihood, 'velocity' : v}))

                if any([any([self.data[x[0]][0] == time     for x in edges]),
                        any([self.data[x[1]][0] == time + 1 for x in edges]),
                        likelihood > self.decision_boundary]):
                    self.graph.add_edges_from(edges)

    def _get_groups(self, start, stop):
        nodes1, nodes2 = [], []
        for node in self.graph.nodes():
            if node in self.special_nodes: continue
            if (time := self.data[node][0]) < start or time > stop: continue
            if list(self.graph._succ[node]) == [] and start <= time <  stop: nodes1.append(node)
            if list(self.graph._pred[node]) == [] and start <  time <= stop: nodes2.append(node)
        return(list(map(self._get_trajectory, nodes1)), list(map(self._get_trajectory, nodes2)))

    def _get_trajectory(self, node0):
        nodes = [node0]
        functions = [lambda x: list(y for y in self.graph._pred[x].keys() if type(y) is not str),
                     lambda x: list(y for y in self.graph._succ[x].keys() if type(y) is not str)]
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
        self.graph.add_nodes_from(self.special_nodes, data = 'mommy calls me speshal', position = 'special education class', params = 'speshal')

        for adress, point in zip(np.array(self.dataset)[:,:2], np.array(self.dataset)):
            node = tuple(map(int, adress))
            self.graph.add_node(node, data = list(point), position = point[2:2+self.dim], params = point[2 + self.dim:])

            for i, special_node in enumerate(self.special_nodes):
                edge = [special_node, node]
                if i: edge.reverse()
                self.graph.add_edge(*tuple(edge), likelihood = float(point[0] == self.time_range[i]), velocity = 1e-12)

        self.data       = nx.get_node_attributes(self.graph, 'data')
        self.position   = nx.get_node_attributes(self.graph, 'position')
        self.params     = nx.get_node_attributes(self.graph, 'params')

    def dump_data(self, sub_folder = None, memory = 15, smallest_trajectories = 1):
        if sub_folder is None:  output_path = self.path + '/Tracer Output'
        else:                   output_path = self.path + '/Tracer Output' + sub_folder
        try: os.makedirs(output_path)
        except: pass

        nx.readwrite.gml.write_gml(self.graph, output_path + '/graph.gml', stringizer = lambda x: str(x))
        interpretation = Graph_interpreter(self.graph, self.special_nodes, self.node_trajectory)
        interpretation.events()
        interpretation.families()

        try: os.mkdir(output_path + '/trajectories')
        except FileExistsError:
            map(os.remove, glob.glob(output_path + '/trajectories/**.csv'))
            map(os.remove, glob.glob(output_path + '/trajectories/**.jpg'))


        cols = ['dt'] + ['d'+x for x in self.columns[2:]] + ['likelihoods']
        for i, track in tqdm(enumerate(interpretation.trajectories), desc = 'Saving trajectories: '):

            fig = plt.figure()
            ax = fig.gca(projection='3d')
            X, Y, Z = track.positions[:,0], track.positions[:,1], track.positions[:,2]
            zeros = np.zeros_like(Z)
            ax.plot(X, Y, Z, label='parametric curve', marker='o', ms=1, mec = 'black')
            ax.plot(X, Y, zeros, label='parametric curve',  ms=1)
            ax.plot(X, zeros + 0.03, Z, label='parametric curve',  ms=1)
            ax.plot(zeros, Y, Z, label='parametric curve',  ms=1)
            ax.set(xlim = (0, 0.09), ylim = (0, 0.03), zlim=(0.015, 0.135))
            plt.tight_layout()
            ax.set_xlabel('X')
            ax.set_ylabel('Y')
            ax.set_zlabel('Z')
            plt.savefig(output_path + '/trajectories/trajectory_%i.jpg'%i)
            plt.close()

            table = pd.DataFrame(data = track.data, columns = self.columns)
            table.to_csv(output_path + '/trajectories/data_%i.csv'%i)

            table = pd.DataFrame(data = track.changes, columns = cols)
            table.to_csv(output_path + '/trajectories/changes_%i.csv'%i)

        with open(output_path + '/trajectories/events.csv', 'w') as file:
            events_str = ''
            for event in interpretation.events: events_str += str(event) + '\n'
            file.write(events_str)

def unzip_images(path):
    imgFromZip  = lambda name: Image.open(io.BytesIO(zp.read(name)))
    with zipfile.ZipFile(path) as zp:
        names = zp.namelist()
        try:    names.remove(*[x for x in names if len(x.split('/')[-1]) == 0 or x.split(',')[-1] == 'ini'])
        except: pass
        names.sort(key = lambda x: int(x.split('/')[-1].split('_')[-1].split('.')[0]))
        images = list(map(imgFromZip, tqdm(names, desc = 'Loading images ')))

    scaler_f = lambda x: (2**-8 * int(np.max(x) >= 256) + int(np.max(x) < 256) + 254 * int(np.max(x) == 1))
    scaler   = scaler_f(np.asarray(images[0], np.uint16))

    mapper = lambda img:  np.repeat((np.asarray(img, np.uint16) * scaler).astype(np.uint8)[:,:,np.newaxis],3,2)
    images = list(map(mapper, tqdm(images, desc = 'Mapping images ')))

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
