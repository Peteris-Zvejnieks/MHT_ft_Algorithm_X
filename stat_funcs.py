import numpy as np
from scipy.stats import norm

class statFunc():
    def __init__(self, f, conditions):
        self.f = f
        self.conditions = conditions

    def __repr__(self):
        return str(self.conditions)

    def __call__(self, Y1, Y2):
        return self.f(Y1, Y2)

    def check_conditions(self, Y1, Y2):
        if len(Y1) > 1: a ='n'
        else: a = len(Y1)
        if len(Y2) > 1: b ='n'
        else: b = len(Y2)
        return (self.conditions[0] == a) and (self.conditions[1] == b)

class movement_likelihood_func(statFunc):
    def __init__(self, sig_displacement, k):
        likelihood_displ = lambda dr, dt  : norm.pdf(dr, 0, sig_displacement * dt)/norm.pdf(0, 0, sig_displacement * dt)
        likelihood_S     = lambda dS, sigS: norm.pdf(dS, 0, sigS)/norm.pdf(0,0,sigS)

        def f(stop, start):
            stop, start = stop[0], start[0]

            t1  , t2    = stop.ending[0], start.beginning[0]
            dt = t2 - t1

            sig_S = (start.sig_S + stop.sig_S)/2
            dS    = start.mu_S - stop.mu_S
            b     = likelihood_S(dS, sig_S)

            try:
                p1 = stop(t1 + dt/2)
                try: p2 = start(t2 - dt/2)
                except:
                    p1 = stop(t2)
                    p2 = start.beginning[2:4]
                finally: a = likelihood_displ(np.linalg.norm(p2 - p1), dt)

            except:
                p1 = stop.ending[2:4]
                try:
                    p2 = start(t1)
                    a = likelihood_displ(np.linalg.norm(p2 - p1), dt)
                except:
                    p2 = start.beginning[2:4]
                    dr      = np.linalg.norm(p2 - p1)
                    mu_d    = (start.mu_V + stop.mu_V)/2 * dt
                    sigma_d = (start.sig_V + stop.sig_V)/2 * dt
                    a       = norm.pdf(dr, mu_d, sigma_d)/norm.pdf(mu_d, mu_d, sigma_d)
            finally: return k * a + (1 - k) * b

        super().__init__(f, [1,1])

class new_or_gone_likelihood_func_Y(statFunc):
    def __init__(self, a, b, c):
        f0 = lambda x: 1/(1+np.exp(a*(x-b)))
        def f(stop, start):
            if c:
                trajectory, dt = start[0], -1
                t = trajectory.beginning[0] + dt
                try:    y = trajectory(t)[1]
                except: y = trajectory.beginning[3] + trajectory.mu_V * dt
            else:
                trajectory, dt = stop[0], 1
                t = trajectory.ending[0] + dt
                try:    y = trajectory(t)[1]
                except: y = trajectory.ending[3]  + trajectory.mu_V * dt
            return f0(y)

        super().__init__(f, [1 - c, c])

class new_or_gone_likelihood_func_X(statFunc):
    def __init__(self, a, b, c):
        f0 = lambda x: 1/(1+np.exp(a*(x-b)))
        def f(stop, start):
            if c:
                trajectory, dt = start[0], -1
                t = trajectory.beginning[0] + dt
                try:    x = trajectory(t)[1]
                except: x = trajectory.beginning[2] + trajectory.mu_V * dt
            else:
                trajectory, dt = stop[0], 1
                t = trajectory.ending[0] + dt
                try:    x = trajectory(t)[1]
                except: x = trajectory.ending[2]  + trajectory.mu_V * dt
            return f0(x)

        super().__init__(f, [1 - c, c])


class multi_bubble_likelihood_func(statFunc):
    def __init__(self, sig_displ, k, c):
        likelihood_displ = lambda p1, p2, dt: np.divide(norm.pdf(np.linalg.norm(p2 - p1), 0, sig_displ*dt),norm.pdf(0, 0, sig_displ*dt))
        likelihood_S     = lambda dS, S_sig: norm.pdf(dS, 0, S_sig)/norm.pdf(0, 0, S_sig)
        f0 = lambda pos, Ss: np.array([np.dot(pos[:,0], Ss**2)/np.sum(Ss**2),
                                       np.dot(pos[:,1], Ss**2)/np.sum(Ss**2)])
        c = int(c)
        def f(stops, starts):
            if c: #split
                trajectory   = stops[0]
                trajectories = starts
                t, p = trajectory.ending[0], trajectory.ending[2:4]
                ts = [traject.beginning[0] for traject in trajectories]
            else: #merge
                trajectories = stops
                trajectory   = starts[0]
                t, p = trajectory.beginning[0], trajectory.beginning[2:4]
                ts = [traject.ending[0] for traject in trajectories]

            positions = []
            dts = []
            Ss = []

            for traject, time in zip(trajectories, ts):
                try:
                    positions.append(traject(t))
                    Ss.append(traject.mu_V)
                    dts.append(abs(time - t))
                except: pass

            if len(positions) != 0:
                p_predict = f0(np.array(positions), np.array(Ss))

                frac    = len(Ss)/len(trajectories)
                a       = likelihood_displ(p_predict, p, np.average(dts)) * frac
            else:
                a=0

            S       = np.sum(np.array([traject.mu_V for traject in trajectories]))
            sig_S   = np.sum(np.array([tr.sig_S for tr in trajectories]))

            dS      = trajectory.mu_S - S
            S_sig   = (trajectory.sig_S + sig_S)/2
            b       = likelihood_S(dS, S_sig)

            return k * a + (1 - k) * b


        super().__init__(f, [['n', 1],[1, 'n']][c])