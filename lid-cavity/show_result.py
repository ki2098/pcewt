from scipy.interpolate import griddata
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

pce_df = pd.read_csv('data/pce-statistics.csv')
mc_df = pd.read_csv('data/mc-statistics.csv')

points = pce_df[['x', 'y']].values

x_plot = 0
y_plot = 0
vy = np.linspace(pce_df['y'].min(), pce_df['y'].max(), 100)
vx = np.full_like(vy, x_plot)
vline = np.column_stack((vx, vy))
hx = np.linspace(pce_df['x'].min(), pce_df['x'].max(), 100)
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
    pce_var = pce_df[varname].values
    pce_var_vline = griddata(points, pce_var, vline)
    pce_var_hline = griddata(points, pce_var, hline)

    mc_var = mc_df[varname].values
    mc_var_vline = griddata(points, mc_var, vline)
    mc_var_hline = griddata(points, mc_var, hline)

    plt.figure()
    plt.plot(vy, pce_var_vline, label=f'PCE (P=5) along x = {x_plot}', linewidth=1, color='blue')
    plt.plot(hx, pce_var_hline, label=f'PCE (P=5) along y = {y_plot}', linewidth=1, color='blue', ls=':')
    plt.scatter(vy[::3], mc_var_vline[::3], label=f'Monte Carlo (500 samples) along x = {x_plot}', marker='+', s=50, c='red')
    plt.scatter(hx[::3], mc_var_hline[::3], label=f'Monte Carlo (500 samples) along y = {y_plot}', marker='.', c='red')
    if varname == 'E[u]':
        plt.scatter(ghia_vy, ghia_u_vline, label=f'Ghia (1982) u along vertical line cross geometric center', marker='D', c='#00c000', s=50)
    elif varname == 'E[v]':
        plt.scatter(ghia_hx, ghia_v_hline, label=f'Ghia (1982) v along horizontal line cross geometric center', marker='D', c='#00c000', s=50)
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

# pce_u_mean = pce_df['u_mean'].values
# pce_u_mean_vline = griddata(points, pce_u_mean, vline)
# pce_u_mean_hline = griddata(points, pce_u_mean, hline)

# mc_u_mean = mc_df['u_mean'].values
# mc_u_mean_vline = griddata(points, mc_u_mean, vline)
# mc_u_mean_hline = griddata(points, )

# plt.figure()
# plt.plot(y_line, pce_u_mean_along_line, label='PCE (P=6)', linewidth=1)
# plt.scatter(y_line[::3], mc_u_mean_along_line[::3], label='Monte Carlo (500 samples)', marker='+', s=50)
# plt.xlabel('y')
# plt.ylabel('E[u]')
# plt.title(f'E[u] along x={x_plot}')
# plt.grid(True)
# plt.legend()

# pce_u_var = pce_df['u_var'].values
# pce_u_var_along_line = griddata(points, pce_u_var, plot_points)

# mc_u_var = mc_df['u_var'].values
# mc_u_var_along_line = griddata(points, mc_u_var, plot_points)

# plt.figure()
# plt.plot(y_line, pce_u_var_along_line, label='PCE (P=6)', linewidth=1, color='red')
# plt.scatter(y_line[::3], mc_u_var_along_line[::3], label='Monte Carlo (500 samples)', marker='+', c='r', s=50)
# plt.xlabel('y')
# plt.ylabel('Var[u]')
# plt.title(f'Var[u] along x={x_plot}')
# plt.grid(True)
# plt.legend()

# pce_v_mean = pce_df['v_mean'].values
# pce_v_mean_along_line = griddata(points, pce_v_mean, plot_points)

# mc_v_mean = mc_df['v_mean'].values
# mc_v_mean_along_line = griddata(points, mc_v_mean, plot_points)

# plt.figure()
# plt.plot(y_line, pce_v_mean_along_line, label='PCE (P=6)', linewidth=1, color='green')
# plt.scatter(y_line[::3], mc_v_mean_along_line[::3], label='Monte Carlo (500 samples)', marker='+', c='g', s=50)
# plt.xlabel('y')
# plt.ylabel('E[v]')
# plt.title(f'E[v] along x={x_plot}')
# plt.grid(True)
# plt.legend()

# pce_v_var = pce_df['v_var'].values
# pce_v_var_along_line = griddata(points, pce_v_var, plot_points)

# mc_v_var = mc_df['v_var'].values
# mc_v_var_along_line = griddata(points, mc_v_var, plot_points)

# plt.figure()
# plt.plot(y_line, pce_v_var_along_line, label='PCE (P=6)', linewidth=1, color='black')
# plt.scatter(y_line[::3], mc_v_var_along_line[::3], label='Monte Carlo (500 samples)', marker='+', c='k', s=50)
# plt.xlabel('y')
# plt.ylabel('Var[v]')
# plt.title(f'Var[v] along x={x_plot}')
# plt.grid(True)
# plt.legend()

# plt.show()
