from scipy.interpolate import griddata
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

ipce_df = pd.read_csv('data/ipce-statistics.csv')
nipce_df = pd.read_csv('data/nipce-quad-statistics.csv')
mc_df = pd.read_csv('data/mc-statistics.csv')

points = ipce_df[['x', 'y']].values

x_plot = 0
y_plot = 0
vy = np.linspace(ipce_df['y'].min(), ipce_df['y'].max(), 100)
vx = np.full_like(vy, x_plot)
vline = np.column_stack((vx, vy))
hx = np.linspace(ipce_df['x'].min(), ipce_df['x'].max(), 100)
hy = np.full_like(vx, y_plot)
hline = np.column_stack((hx, hy))

ghia_vy = [
    gy - 0.5 for gy in [
    1.0,
    0.9766,
    0.9531,
    0.8516,
    0.7344,
    0.6172,
    0.5000,
    0.4531,
    0.2813,
    0.1719,
    0.0625,
    0.0000
]]

ghia_u_vline = [
    1, 
    0.75837,
    0.55892,
    0.29093,
    0.16256,
    0.02135,
    -0.11477,
    -0.17119,
    -0.32726,
    -0.24299,
    -0.09266,
    0.00000
]

ghia_hx = [
    gx - 0.5 for gx in [
    1.0000,
    0.9688,
    0.9609,
    0.9531,
    0.9453,
    0.8594,
    0.8047,
    0.5000,
    0.2344,
    0.1563,
    0.0938,
    0.0625,
    0.0000
]]

ghia_v_hline = [
    0.00000,
    -0.12146,
    -0.15663,
    -0.19254,
    -0.22847,
    -0.44993,
    -0.38598,
    0.05188,
    0.30174,
    0.28124,
    0.22965,
    0.18360,
    0.00000 
]

def plot_variable(varname: str):
    ipce_var = ipce_df[varname].values
    ipce_var_vline = griddata(points, ipce_var, vline)
    ipce_var_hline = griddata(points, ipce_var, hline)

    nipce_var = nipce_df[varname].values
    nipce_var_vline = griddata(points, nipce_var, vline)
    nipce_var_hline = griddata(points, nipce_var, hline)

    mc_var = mc_df[varname].values
    mc_var_vline = griddata(points, mc_var, vline)
    mc_var_hline = griddata(points, mc_var, hline)

    plt.figure()
    plt.plot(vy, ipce_var_vline, label=f'iPCE (P=5) along x = {x_plot}', linewidth=1, color='blue')
    plt.plot(hx, ipce_var_hline, label=f'iPCE (P=5) along y = {y_plot}', linewidth=1, color='blue', ls=':')
    plt.scatter(vy[::3], nipce_var_vline[::3], label=f'niPCE (6 samples) along x = {x_plot}', marker='+', s=50, c='red')
    plt.scatter(hx[::3], nipce_var_hline[::3], label=f'niPCE (6 samples) along y = {y_plot}', marker='.', c='red')
    plt.plot(vy, mc_var_vline, label=f'Monte Carlo (100 samples) along x = {x_plot}', linewidth=1, color='black', alpha=0.5)
    plt.plot(hx, mc_var_hline, label=f'Monte Carlo (100 samples) along y = {y_plot}', linewidth=1, color='black', ls=':', alpha=0.5)
    if varname == 'E[u]':
        plt.scatter(ghia_vy, ghia_u_vline, label=f'Ghia (1982)', marker='D', c='#00c000', s=25)
    elif varname == 'E[v]':
        plt.scatter(ghia_hx, ghia_v_hline, label=f'Ghia (1982)', marker='D', c='#00c000', s=25)
    plt.xlabel('x or y')
    plt.ylabel(varname)
    plt.title(f'{varname} along line x={x_plot} and line y={y_plot}')
    plt.grid(True)
    plt.legend()
    plt.savefig(f'data/{varname}.png')

plot_variable('E[u]')
plot_variable('Var[u]')
plot_variable('E[v]')
plot_variable('Var[v]')

