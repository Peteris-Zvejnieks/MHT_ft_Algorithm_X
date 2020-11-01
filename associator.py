import numpy as np
import itertools as itt

class Association_condition():
    def __init__(self, f):
        self.f = f
    def __call__(self, Y1, Y2):
        return self.f(Y1, Y2)

class asc_condition(Association_condition):
    def __init__(self,
                 max_displ_per_frame = 45,
                 radius_multiplyer = 2.5,
                 min_displacement = 30):

        def f(stop, start):
            if stop == start:                                                           return False

            dt = start.beginning[0] - stop.ending[0]
            dr = np.linalg.norm(start.beginning[2:4] - stop.ending[2:4])

            if   dt <= 0:                                                               return False
            elif dr > max_displ_per_frame * dt:                                         return False
            elif dr > (stop.ending[4]+start.beginning[4])/2 * radius_multiplyer * dt:   return False
            if dr < min_displacement * dt:                                              return True
            else:                                                                       return True

        super().__init__(f)

class asc_condition_particles(Association_condition):
    def __init__(self,
                 max_displ_per_frame = 45,
                 radius_multiplyer = 3,
                 min_displacement = 6):

        def f(stop, start):
            if stop == start:                                                               return False

            dt = start.beginning[0] - stop.ending[0]
            dr = np.linalg.norm(start.beginning[2:4] - stop.ending[2:4])

            if   dt <= 0:                                                                   return False
            if dr > max_displ_per_frame * dt:                                               return False
            if dr > ((stop.ending[5]+start.beginning[5])/2)**0.5 * radius_multiplyer * dt:  return False
            if dr < min_displacement * dt:                                                  return True
            else:                                                                           return True

        super().__init__(f)

class asc_condition_3D_bubbles(Association_condition):
    def __init__(self, max_displ_per_frame, radius_multiplyer, min_displacement):
        def f(stop, start):
            if stop == start: return False
            dt = start.beginning[0] - stop.ending[0]
            dr = np.linalg.norm(start.beginning[2:4] - stop.ending[2:4])
            R1, R2 = (3*np.pi*stop.mu_S/4)**(1/3), (3*np.pi*start.mu_S/4)**(1/3)

            if dt <= 0: return False
            if dr > max_displ_per_frame * dt:                                               return False
            if dr > (R1+R2)/2 * radius_multiplyer * dt:                                     return False
            if dr < min_displacement * dt:                                                  return True
            else:                                                                           return True

        super().__init__(f)

class Combination_constraint():
    def __init__(self, f):
        self.f = f
    def __call__(self, Y1, Y2):
        return self.f(Y1, Y2)

class comb_constr(Combination_constraint):
    def __init__(self, upsilon, v_scaler = 10, max_a = 5):
        d_fi = lambda u, v: np.arccos(np.dot(u, v)/(np.linalg.norm(u)*np.linalg.norm(v)))
        def f(stops, starts):
            #If new or gone - defaults to True
            if len(stops) == 0 or len(starts) == 0: return True
            #Filters out quick changes in direction depending on velocity
            if len(stops) == 1 and len(starts) == 1:
                stop, start = stops[0], starts[0]
                dt = start.beginning[0] - stop.ending[0]
                mid_v = (start.beginning[2:4] - stop.ending[2:4])/dt
                if len(stop)  >= 2:
                    v = stop.displacements[-1,:]/stop.changes[-1,0]
                    acc = 2 * (mid_v - v)/(stop.changes[-1,0] + dt)
                    if np.linalg.norm(acc) > max_a: return False
                    if d_fi(v, mid_v) > (np.pi + 1e-6) * np.exp(-np.linalg.norm(v)/v_scaler): return False
                if len(start) >= 2:
                    v = start.displacements[0,:]/start.changes[0,0]
                    acc = 2 * (v - mid_v)/(start.changes[0,0] + dt)
                    if np.linalg.norm(acc) > max_a: return False
                    if d_fi(mid_v, v) > (np.pi + 1e-3) * np.exp(-np.linalg.norm(v)/v_scaler): return False
            #Area check
            S1, S2, sigs = 0, 0, 0
            for stop in stops:
                S1   += stop.mu_S
                sigs += stop.sig_S / stop.mu_S
            for start in starts:
                S2   += start.mu_S
                sigs += start.sig_S / start.mu_S
            sigs     /= len(stops) + len(starts)
            if abs(S2 - S1)/max(S2, S1) < upsilon * sigs:
                return True
            else: return False

        super().__init__(f)

'''
Given two lists of objects and an fittingly defined pairwise association condition will generate
all the associations and combinations of them.
'''
class Associator():
    def __init__(self, association_condition,
                 combination_constraint, /,
                 max_k = 3, for_optimizer = True):
        self.association_condition   = association_condition
        self.combination_constraint  = combination_constraint
        self.max_k          = max_k
        self.for_optimizer  = for_optimizer

    def __call__(self, group1, group2):
        all_associations1, all_associations2 = [], []

        self.offset = len(group1)
        for i, obj in enumerate(group1):
            associables     = [j for j, y in enumerate(group2) if self.association_condition(obj, y)]
            combinations    = self._getAllCombinations(associables, 'Exit')
            all_associations2.extend(combinations)
            all_associations1.extend([(i,) for x in combinations])

        for j, obj in enumerate(group2):
            associables     = [i for i, x in enumerate(group1) if self.association_condition(x, obj)]
            combinations    = self._getAllCombinations(associables, 'Entry', 2)
            all_associations1.extend(combinations)
            all_associations2.extend([(j,) for x in combinations])

        all_Y1 = [self._map_adresses_to_data(group1, asc) for asc in all_associations1]
        all_Y2 = [self._map_adresses_to_data(group2, asc) for asc in all_associations2]
        Ys, ascs = tuple(), tuple()
        for Y1, Y2, asc1, asc2 in zip(all_Y1, all_Y2, all_associations1, all_associations2):
            if self.combination_constraint(Y1, Y2):
                Ys   += ((Y1, Y2),)
                ascs += ((asc1, asc2),)
        associations_for_optimizer = list(map(self._for_optimizer, ascs))

        if self.for_optimizer: return (associations_for_optimizer, Ys, ascs)
        else: return (Ys, ascs)

    def _getAllCombinations(self, things, none_str, min_k = 1):
        assert (type(none_str) is str), 'Incorrect type for none_str'
        combinations = [(none_str,)]
        for k in range(min_k, min(len(things), self.max_k) + 1):
            combinations.extend(list(itt.combinations(things, k)))
        return combinations

    def _map_adresses_to_data(self, group, asc):
        try:    return [group[i] for i in asc]
        except TypeError: return []

    def _for_optimizer(self, x):
        a, b = x
        f0 = lambda x: tuple(map(lambda y: y + self.offset, x))
        if   type(a[0]) is str : return f0(b)
        elif type(b[0]) is str : return a
        else                   : return a + f0(b)
