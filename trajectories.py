from scipy import interpolate as interp
import networkx as nx
import numpy as np

class node_trajectory_base():
    class ExtrapolationError(Exception): pass

    def __init__(self, graph):
        self.backbone = graph
        self.nodes = list(graph.nodes)
        self.nodes.sort(key = lambda x: x[0])
        self._get_data()
        self._splinify()

    def __repr__(self): return str(self.time)
    def __len__(self):  return len(self.nodes)

    def __call__(self, t):
        try: return self.extrapolate(t)
        except AttributeError: raise node_trajectory_base.ExtrapolationError('No method for extrapolation has been set')

    def _get_data(self):
        data = nx.get_node_attributes(self.backbone, 'data')
        positions = nx.get_node_attributes(self.backbone, 'position')
        params = nx.get_node_attributes(self.backbone, 'params')
        self.data       = np.array(list(map(lambda node: data[node], self.nodes)))
        self.positions  = np.array(list(map(lambda node: positions[node], self.nodes)))
        self.params    = np.array(list(map(lambda node: params[node], self.nodes)))
        self.beginning, self.ending = self.data[0], self.data[-1]
        self.time = self.data[:,0]

        likelihoods = nx.get_edge_attributes(self.backbone, 'likelihood')
        self.likelihoods = np.array(list(map(lambda x: likelihoods[x], zip(self.nodes[:-1], self.nodes[1:]))))
        velocities = nx.get_edge_attributes(self.backbone, 'velocity')
        self.velocities = np.array(list(map(lambda x: velocities[x], zip(self.nodes[:-1], self.nodes[1:]))))

    def interpolate(self, t, der = 0): return np.array(interp.splev(t, self.tck, der = der))

    def _splinify(self):
        if self.data.shape[0] == 1: pass
        else:
            if self.data.shape[0] <= 3: k = 1
            else:                       k = 3
            points = [self.positions[:,i] for i in range(self.positions.shape[1])]
            self.tck, self.u = interp.splprep(points, u = self.time, k = k, s = 8e2)

class node_trajectory(node_trajectory_base):
    def __init__(self, graph):
        super().__init__(graph)
        self._get_changes()
        self._get_acceleration()

    def extrapolate(self, t):
        time = self.time[-int(t > self.time[0])]
        a = 0.5
        dt = t - time
        return self.interpolate(time) + self.interpolate(t - a * dt/abs(dt), 1) * dt

    def _get_changes(self):
        self.changes = np.zeros((max(len(self) - 1, 1), self.data.shape[1]))
        self.changes[0,0] = 1
        self.displacements = self.positions[1:,:] - self.positions[:-1,:]
        if len(self) > 1:
            self.changes[:, 0]  = self.data[1:,0 ] - self.data[:-1,0 ]
            self.changes[:, 1:1 + self.positions.shape[1]]  = self.displacements
            self.changes[:, 1 + self.positions.shape[1]:-1] = self.params[1:,:] - self.params[:-1,:]
            self.changes[:, -1] = self.likelihoods.ravel()

    def _get_acceleration(self):
        self.acceleration = np.zeros((max(len(self) - 2, 1), 2))
        if len(self) > 2:
            norms = np.linalg.norm(self.displacements, axis = 1)
            dot_products = np.array(list(map(lambda x: np.dot(x[0], x[1]), zip(self.displacements[1:,:], self.displacements[:-1,:]))))
            self.acceleration[:,0] = np.arccos(dot_products/(norms[1:] * norms[:-1]))/self.changes[1:,0]
            self.acceleration[:,1] = (norms[1:] - norms[:-1])/self.changes[1:,0]

    def _get_stats(self):
        velocities   = np.linalg.norm(self.displacements, axis = 1)/self.changes[:,0]
        self.mu_V    = np.average(velocities)
        self.sig_V   = np.std(velocities)
        self.mu_S    = np.average((S := self.params[:,0]))
        self.sig_S   = np.std(S)

class node_trajectory_with_stats():
    def __init__(self, mu_V0, sig_V0, r_sig_S0):
        self.mu_V0, self.sig_V0, self.r_sig_S0 = mu_V0, sig_V0, r_sig_S0

    def __call__(self, graph):
        trajectory = node_trajectory(graph)
        if len(trajectory) <= 2:
            trajectory.mu_V    = self.mu_V0
            trajectory.sig_V   = self.sig_V0
            trajectory.mu_S    = np.average(trajectory.params[:,0])
            trajectory.sig_S   = trajectory.mu_S * self.r_sig_S0
        else: trajectory._get_stats()
        return trajectory
