from scipy.interpolate import griddata
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

pce_df = pd.read_csv('data/pce-statistics.csv')
mc_df = pd.read_csv('data/mc-statistics.csv')

points = pce_df[['x', 'y']].values

x_plot = 0.0
y_line = np.linspace(pce_df['y'].min(), pce_df['y'].max(), 100)
x_line = np.full_like(y_line, x_plot)
plot_points = np.column_stack((x_line, y_line))

pce_u_mean = pce_df['u_mean'].values
pce_u_mean_along_line = griddata(points, pce_u_mean, plot_points)

mc_u_mean = mc_df['u_mean'].values
mc_u_mean_along_line = griddata(points, mc_u_mean, plot_points)

plt.figure()
plt.plot(y_line, pce_u_mean_along_line, label='PCE (P=6)', linewidth=1)
plt.scatter(y_line[::3], mc_u_mean_along_line[::3], label='Monte Carlo (500 samples)', marker='+', s=50)
plt.xlabel('y')
plt.ylabel('E[u]')
plt.title(f'E[u] along x={x_plot}')
plt.grid(True)
plt.legend()

pce_u_var = pce_df['u_var'].values
pce_u_var_along_line = griddata(points, pce_u_var, plot_points)

mc_u_var = mc_df['u_var'].values
mc_u_var_along_line = griddata(points, mc_u_var, plot_points)

plt.figure()
plt.plot(y_line, pce_u_var_along_line, label='PCE (P=6)', linewidth=1, color='red')
plt.scatter(y_line[::3], mc_u_var_along_line[::3], label='Monte Carlo (500 samples)', marker='+', c='r', s=50)
plt.xlabel('y')
plt.ylabel('Var[u]')
plt.title(f'Var[u] along x={x_plot}')
plt.grid(True)
plt.legend()

pce_v_mean = pce_df['v_mean'].values
pce_v_mean_along_line = griddata(points, pce_v_mean, plot_points)

mc_v_mean = mc_df['v_mean'].values
mc_v_mean_along_line = griddata(points, mc_v_mean, plot_points)

plt.figure()
plt.plot(y_line, pce_v_mean_along_line, label='PCE (P=6)', linewidth=1, color='green')
plt.scatter(y_line[::3], mc_v_mean_along_line[::3], label='Monte Carlo (500 samples)', marker='+', c='g', s=50)
plt.xlabel('y')
plt.ylabel('E[v]')
plt.title(f'E[v] along x={x_plot}')
plt.grid(True)
plt.legend()

pce_v_var = pce_df['v_var'].values
pce_v_var_along_line = griddata(points, pce_v_var, plot_points)

mc_v_var = mc_df['v_var'].values
mc_v_var_along_line = griddata(points, mc_v_var, plot_points)

plt.figure()
plt.plot(y_line, pce_v_var_along_line, label='PCE (P=6)', linewidth=1, color='black')
plt.scatter(y_line[::3], mc_v_var_along_line[::3], label='Monte Carlo (500 samples)', marker='+', c='k', s=50)
plt.xlabel('y')
plt.ylabel('Var[v]')
plt.title(f'Var[v] along x={x_plot}')
plt.grid(True)
plt.legend()

plt.show()
