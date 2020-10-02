from scipy import interpolate as interp
from scipy.stats import norm
import itertools as itt
import networkx as nx
import numpy as np
axis = np.newaxis

class Trajectory_stats():
    def __init__(self, get_stats, extrapolate_stats):
        self.stats = []
        self.f1 = get_stats
        self.f2 = extrapolate_stats

    def __call__(self, trajectory):
        try:    self.stats.append(stats := self.f1(trajectory))
        except:      stats = self.f2(self.stats, trajectory)
        return       stats

class trajectory_stats_v1(Trajectory_stats):
    def __init__(self, mu_D0, sig_D0, r_sig_S0):
        def get_stats(trajectory):
            if len(trajectory) <= 2: raise TypeError('Too short of a trajectory')

            velocities = np.linalg.norm(trajectory.changes[:,1:3], axis = 1)/trajectory.changes[:,0]
            mu_V    = np.average(velocities)
            sig_V   = np.std(velocities)

            mu_S    = np.average((S := trajectory.data[:,5]))
            sig_S   = np.std(S)

            return {'mu_D'      : mu_V,
                    'sig_D'     : sig_V,
                    'mu_S'      : mu_S,
                    'sig_S'     : sig_S}

        def extrapolate(stats, trajectory):
            data = trajectory.data
            #if len(stats) == 0:
            return {'mu_D' : mu_D0,
                    'sig_D': sig_D0,
                    'mu_S' : float(np.average(data[:,5])),
                    'sig_S': r_sig_S0 * float(np.average(data[:,5]))}

            mu_D  = np.average(np.array([x['mu_D']  for x in stats]))
            sig_D = np.average(np.array([x['sig_D'] for x in stats]))
            mu_S  = np.average(data[:,5])
            sig_S = mu_S * np.average(np.array([x['sig_S']/x['mu_S'] for x in stats]))

            return {'mu_D'      : mu_D,
                    'sig_D'     : sig_D,
                    'mu_S'      : mu_S,
                    'sig_S'     : sig_S}

        super().__init__(get_stats, extrapolate)

class Node_trajectory():
    def __init__(self, graph):
        self.backbone = graph
        self.nodes = list(graph.nodes)
        self.nodes.sort(key = lambda x: x[0])
        self._get_data()
        self._splinify()
        #self._get_outliers()

    def __call__(self, time, der = 0):
        if self.time[0] <= time <= self.time[-1]:
            return np.array(interp.splev(time, self.tck, der = der))
        else:
            t = self.time[-int(time > self.time[0])]
            a = 0.5
            dt = time - t
            return self(t, der) + self(t - a * dt/abs(dt), der + 1) * dt

    def __repr__(self): return str(self.time)
    def __len__(self):  return len(self.nodes)

    def get_stats(self, trajectory_stats):
        self.stats = trajectory_stats(self)

    def _splinify(self):
        if self.data.shape[0] == 1: pass
        else:
            if self.data.shape[0] <= 3: k = 1
            else:                       k = 3
            points = [self.data[:,2], self.data[:,3]]
            self.tck, self.u = interp.splprep(points, u = self.time, k = k, s = 8e2)

    def _get_data(self):
        data = nx.get_node_attributes(self.backbone, 'data')
        self.data = np.array(list(map(lambda node: data[node], self.nodes)))
        self.beginning, self.ending = self.data[0], self.data[-1]
        self.time = self.data[:,0]

        likelihoods = nx.get_edge_attributes(self.backbone, 'likelihood')
        self.likelihoods = np.array(list(map(lambda x: likelihoods[x], zip(self.nodes[:-1], self.nodes[1:]))))

        self.changes = np.zeros((max(len(self) - 1, 1), self.data.shape[1]))
        self.changes[0,0] = 1
        if len(self) > 1:
            self.changes[:,   0] = self.data[1:,0 ] - self.data[:-1,0 ]
            self.changes[:,1:-1] = self.data[1:,2:] - self.data[:-1,2:]
            self.changes[:,  -1] = self.likelihoods.ravel()

        self.acceleration = np.zeros((max(len(self) - 2, 1), 2))
        if len(self) > 2:
            displ = self.changes[:,1:3]
            norms = np.linalg.norm(displ, axis = 1)
            self.acceleration[:,0] = np.arccos((displ[1:,0]*displ[:-1,0] + displ[1:,1]*displ[:-1,1])/(norms[1:] * norms[:-1]))/self.changes[1:,0]
            self.acceleration[:,1] = (norms[1:] - norms[:-1])/self.changes[1:,0]

    # def _get_outliers(self):
    #     sigmas = 2
    #     deltas = np.einsum('ij, i -> ij', self.changes[:,1:-1], self.changes[:,0]**-1)

class Trajectories(list):
    def __init__(self, graph, special_nodes):
        self.digraph = graph
        self.graph = graph.to_undirected()
        self.special_nodes = special_nodes

        self._get_trajectories()
        self._get_events()
        self._get_families()

    def _get_trajectories(self):
        tmp_digraph = self.digraph.copy()
        tmp_digraph.remove_nodes_from(self.special_nodes)
        for node in tmp_digraph.nodes:
            if len(ins  := list(tmp_digraph.in_edges(node)))  > 1:
                tmp_digraph.remove_edges_from(ins)
            if len(outs := list(tmp_digraph.out_edges(node))) > 1:
                tmp_digraph.remove_edges_from(outs)

        tmp_graph = tmp_digraph.to_undirected()

        self.paths = list(tmp_digraph.subgraph(c) for c in nx.connected_components(tmp_graph))
        self.paths.sort(key = lambda x: -len(x.nodes))
        super().__init__(map(Node_trajectory, self.paths))

    def _find_by_node(self, node):
        for i, x in enumerate(self):
            if node in x.nodes: return i

    def _get_events(self):
        self.events = []
        likelihoods = nx.get_edge_attributes(self.digraph, 'likelihood')

        for node in self.digraph.nodes:
            if len(ins  := list(self.digraph.in_edges(node)))  > 1:
                if node in self.special_nodes:
                    for iny in ins:
                        self.events.append([[self._find_by_node(iny[0])],
                                            [node],
                                            likelihoods[iny]])
                else:
                    self.events.append([[self._find_by_node(iny[0]) for iny in ins],
                                        [self._find_by_node(node)],
                                        likelihoods[ins[0]]])

            if len(outs := list(self.digraph.out_edges(node))) > 1:
                if node in self.special_nodes:
                    for outy in outs:
                        self.events.append([[node],
                                            [self._find_by_node(outy[1])],
                                            likelihoods[outy]])
                else:
                    self.events.append([[self._find_by_node(node)],
                                        [self._find_by_node(outy[1]) for outy in outs],
                                        likelihoods[outs[0]]])

    def _get_families(self):
        tmp_graph = self.graph.copy()
        tmp_graph.remove_nodes_from(self.special_nodes)
        self.families = list(self.digraph.subgraph(c.union(set(self.special_nodes))) for c in nx.connected_components(tmp_graph))
        self.families.sort(key = lambda graph: -len(list(graph.nodes)))
