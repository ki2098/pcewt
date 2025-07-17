import numpy as np
import json

def read_M(P, fname):
    with open(fname) as f:
        M_non_zero = json.load(f)
        M = np.zeros((P + 1, P + 1, P + 1))
        for a, b, g, m in M_non_zero:
            M[a, b, g] = m
        return M

def compute_utopia_convection(ww, w, c, e, ee, u, dx):
    return (u*(- ee + 8*e - 8*w + ww) + abs(u)*(ee - 4*e + 6*c - 4*w + ww))/(12*dx)

def compute_convection_at_cell(g, U : np.ndarray, M : np.ndarray, dx, i, j):
    convection = 0
    for a in range(M.shape[0]):
        u = U[a, i, j, 0]
        v = U[a, i, j, 1]
        for b in range(M.shape[1]):
            Uc  = U[b, i, j]
            Ue  = U[b, i + 1, j]
            Uee = U[b, i + 2, j]
            Uw  = U[b, i - 1, j]
            Uww = U[b, i - 2, j]
            Un  = U[b, i, j + 1]
            Unn = U[b, i, j + 2]
            Us  = U[b, i, j - 1]
            Uss = U[b, i, j - 2]
            m = M[a, b, g]
            convection += m*compute_utopia_convection(Uww, Uw, Uc, Ue, Uee, u, dx)
            convection += m*compute_utopia_convection(Uss, Us, Uc, Un, Unn, v, dx)
    return convection

def compute_diffusion_at_cell(U : np.ndarray, viscosity, dx, i, j):
    Uc  = U[i, j]
    Ue  = U[i + 1, j]
    Uw  = U[i - 1, j]
    Un  = U[i, j + 1]
    Us  = U[i, j - 1]
    return viscosity*((Ue - 2*Uc + Uw) + (Un - 2*Uc + Us))/(dx**2)

def compute_pseudo_U(U : np.ndarray, Ut : np.ndarray, M : np.ndarray, viscosity, dx, dt, gc):
    for g in range(M.shape[2]):
        for i in range(gc, U.shape[0] - gc):
            for j in range(gc, U.shape[1] - gc):
                Uc = U[g, i, j]
                Ut[g, i, j] = Uc \
                    + dt*(- compute_convection_at_cell(g, U, M, dx, i, j) + compute_diffusion_at_cell(U[g], viscosity, dx, i, j))

def compute_pressure_rhs(U : np.ndarray, b : np.ndarray, M : np.ndarray, dx, dt, gc):
    for g in range(M.shape[2]):
        for i in range(gc, U.shape[0] - gc):
            for j in range(gc, U.shape[1] - gc):
                ue = U[g, i + 1, j, 0]
                uw = U[g, i - 1, j, 0]
                vn = U[g, i, j + 1, 1]
                vs = U[g, i, j - 1, 1]
                b[g, i, j] = (ue - uw + vn - vs)/(2*dx*dt)
