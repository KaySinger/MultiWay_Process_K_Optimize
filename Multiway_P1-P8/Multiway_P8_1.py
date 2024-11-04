import numpy as np
import math, random
import matplotlib
matplotlib.use('TkAgg')
from scipy.optimize import minimize
from scipy.integrate import odeint
from Diffusions import Diffusion_Special_P1
import Picture

# 正态分布模拟，得到的结果用于物质稳态浓度
def simulate_normal_distribution(mu, sigma, total_concentration, scale_factor):
    x_values = np.arange(1, 41)
    concentrations = np.exp(-0.5 * ((x_values - mu) / sigma) ** 2) / (sigma * np.sqrt(2 * np.pi))
    concentrations /= sum(concentrations)
    concentrations *= scale_factor
    return concentrations, x_values

# 初始化 k 和 k_inv 数组
def initialize_k_values(concentrations):
    k = np.zeros(257)
    k_inv = np.zeros(256)
    # P1参与后续反应的初始值猜测
    k[0] = 3
    for i in range(1, 40):
        k[i] = 1.5 + random.uniform(0.5, 1) * i
    k_inv[0] = (k[1] * concentrations[0] ** 2) / concentrations[1]
    for i in range(1, 39):
        k_inv[i] = k[i + 1] * concentrations[0] * concentrations[i] / concentrations[i + 1]
    # P2参与后续反应的初始值猜测
    for i in range(37):
        k[i+40] = 1.2 + random.uniform(0.3, 0.5) * i
    k_inv[39] = (k[40] * concentrations[1]**2) / concentrations[3]
    for i in range(36):
        k_inv[i+40] = k[i+41] * concentrations[1] * concentrations[i+2] / concentrations[i+4]
    # P3参与后续反应的初始值猜测
    for i in range(35):
        k[i+77] = 1 + random.uniform(0.3, 0.5) * i
    k_inv[76] = (k[77] * concentrations[2]**2) / concentrations[5]
    for i in range(34):
        k_inv[i+77] = k[i+78] * concentrations[2] * concentrations[i+3] / concentrations[i+6]
    # P4参与后续反应的初始值猜测
    for i in range(33):
        k[i+112] = 1 + random.uniform(0.3, 0.5) * i
    k_inv[111] = (k[112] * concentrations[3]**2) / concentrations[7]
    for i in range(32):
        k_inv[i+112] = k[i+113] * concentrations[3] * concentrations[i+4] / concentrations[i+8]
    # P5参与后续反应的初始值猜测
    for i in range(31):
        k[i + 145] = 1 + random.uniform(0.3, 0.5) * i
    k_inv[144] = (k[145] * concentrations[4] ** 2) / concentrations[9]
    for i in range(30):
        k_inv[i + 145] = k[i + 146] * concentrations[4] * concentrations[i + 5] / concentrations[i + 10]
    # P6参与后续反应的初始值猜测
    for i in range(29):
        k[i + 176] = 1 + random.uniform(0.3, 0.5) * i
    k_inv[175] = (k[176] * concentrations[5] ** 2) / concentrations[11]
    for i in range(28):
        k_inv[i + 176] = k[i + 177] * concentrations[5] * concentrations[i + 6] / concentrations[i + 12]
    # P7参与后续反应的初始值猜测
    for i in range(27):
        k[i + 205] = 1 + random.uniform(0.3, 0.5) * i
    k_inv[204] = (k[205] * concentrations[6] ** 2) / concentrations[13]
    for i in range(26):
        k_inv[i + 205] = k[i + 206] * concentrations[6] * concentrations[i + 7] / concentrations[i + 14]
    # P8参与后续反应的初始值猜测
    for i in range(25):
        k[i + 232] = 1 + random.uniform(0.3, 0.5) * i
    k_inv[231] = (k[232] * concentrations[7] ** 2) / concentrations[15]
    for i in range(24):
        k_inv[i + 232] = k[i + 233] * concentrations[7] * concentrations[i + 8] / concentrations[i + 16]
    return list(k), list(k_inv)

# 定义总的微分方程
def equations(p, t, k, k_inv):
    dpdt_process1 = Diffusion_Special_P1.equations_process1(p, t, k, k_inv)
    dpdt_process2 = Diffusion_Special_P1.equations_process2(p, t, k, k_inv)
    dpdt_process3 = Diffusion_Special_P1.equations_process3(p, t, k, k_inv)
    dpdt_process4 = Diffusion_Special_P1.equations_process4(p, t, k, k_inv)
    dpdt_process5 = Diffusion_Special_P1.equations_process5(p, t, k, k_inv)
    dpdt_process6 = Diffusion_Special_P1.equations_process6(p, t, k, k_inv)
    dpdt_process7 = Diffusion_Special_P1.equations_process7(p, t, k, k_inv)
    dpdt_process8 = Diffusion_Special_P1.equations_process8(p, t, k, k_inv)
    dpdt = [dpdt_process1[i] + dpdt_process2[i] + dpdt_process3[i] + dpdt_process4[i] + dpdt_process5[i] + dpdt_process6[i] + dpdt_process7[i] + dpdt_process8[i] for i in range(41)]
    return dpdt

# 定义目标函数
def objective(params):
    k = params[:257]
    k_inv = params[257:]
    initial_conditions = [10] + [0] * 40
    t = np.linspace(0, 20, 1000)
    sol = odeint(equations, initial_conditions, t, args=(k, k_inv))
    final_concentrations = sol[-1, :]
    target_concentrations = [0] + list(concentrations)
    return np.sum((final_concentrations - target_concentrations) ** 2)

# 回调函数
def callback(xk):
    current_value = objective(xk)
    objective_values.append(current_value)
    if len(objective_values) > 1:
        change = np.abs(objective_values[-1] - objective_values[-2])
        print(f"迭代次数 {len(objective_values) - 1}: 变化 = {change}")

# 模拟正态分布
mu = 20.5
sigma = 10
scale_factor = 10
concentrations, x_values = simulate_normal_distribution(mu, sigma, total_concentration=1.0, scale_factor=scale_factor)
print("理想稳态浓度分布", {f'P{i}': c for i, c in enumerate(concentrations, start=1)})

# 初始K值猜测'
k_initial, k_inv_initial = initialize_k_values(concentrations)
initial_guess = k_initial + k_inv_initial

# 添加参数约束，确保所有k值都是非负的
bounds = [(0, None)] * len(k_initial) + [(0, None)] * len(k_inv_initial)  # 确保长度为 13

# 记录目标函数值
objective_values = []

# 第一次优化
result = minimize(objective, initial_guess, method='L-BFGS-B', bounds=bounds, callback=callback)
k_optimized = result.x[:257]
k_inv_optimized = result.x[257:]
final_precision = result.fun
print(f"优化的最终精度是{final_precision}")

# 输出线程1优化结果
k_result = {f"k{i}": c for i, c in enumerate(k_optimized[:40], start=0)}
k_inv_result = {f"k{i}_inv": c for i, c in enumerate(k_inv_optimized[:39], start=1)}
print("进程1反应式的k:", k_result)
print("进程1反应式的k_inv:", k_inv_result)

# 输出线程2优化结果
k_result = {f"k{i}": c for i, c in enumerate(k_optimized[40:77], start=0)}
k_inv_result = {f"k{i}_inv": c for i, c in enumerate(k_inv_optimized[39:76], start=0)}
print("进程2反应式的k:", k_result)
print("进程2反应式的k_inv:", k_inv_result)

# 输出线程3优化结果
k_result = {f"k{i}": c for i, c in enumerate(k_optimized[77:112], start=0)}
k_inv_result = {f"k{i}_inv": c for i, c in enumerate(k_inv_optimized[76:111], start=0)}
print("进程3反应式的k:", k_result)
print("进程3反应式的k_inv:", k_inv_result)

# 输出线程4优化结果
k_result = {f"k{i}": c for i, c in enumerate(k_optimized[112:145], start=0)}
k_inv_result = {f"k{i}_inv": c for i, c in enumerate(k_inv_optimized[111:144], start=0)}
print("进程4反应式的k:", k_result)
print("进程4反应式的k_inv:", k_inv_result)

# 输出线程5优化结果
k_result = {f"k{i}": c for i, c in enumerate(k_optimized[145:176], start=0)}
k_inv_result = {f"k{i}_inv": c for i, c in enumerate(k_inv_optimized[144:175], start=0)}
print("进程5反应式的k:", k_result)
print("进程5反应式的k_inv:", k_inv_result)

# 输出线程6优化结果
k_result = {f"k{i}": c for i, c in enumerate(k_optimized[176:205], start=0)}
k_inv_result = {f"k{i}_inv": c for i, c in enumerate(k_inv_optimized[175:204], start=0)}
print("进程6反应式的k:", k_result)
print("进程6反应式的k_inv:", k_inv_result)

# 输出线程7优化结果
k_result = {f"k{i}": c for i, c in enumerate(k_optimized[205:232], start=0)}
k_inv_result = {f"k{i}_inv": c for i, c in enumerate(k_inv_optimized[204:231], start=0)}
print("进程7反应式的k:", k_result)
print("进程7反应式的k_inv:", k_inv_result)

# 输出线程8优化结果
k_result = {f"k{i}": c for i, c in enumerate(k_optimized[232:257], start=0)}
k_inv_result = {f"k{i}_inv": c for i, c in enumerate(k_inv_optimized[231:], start=0)}
print("进程8反应式的k:", k_result)
print("进程8反应式的k_inv:", k_inv_result)

# 利用优化后的参数进行模拟
initial_conditions = [10] + [0] * 40
t = np.linspace(0, 20, 1000)
sol = odeint(equations, initial_conditions, t, args=(k_optimized, k_inv_optimized))

Deviation = [0] * 40
Error = [0] * 40
p = list(concentrations)
for i in range(40):
    Deviation[i] = p[i] - sol[-1][i+1]
    if p[i] != 0:
        Error[i] = Deviation[i] / p[i]
    else:
        Error[i] = float('inf')

deviations = {f'P{i}': c for i, c in enumerate(Deviation, start=1)}
Error_Ratio = {f'Error Ratio P{i}': c for i, c in enumerate(Error, start=1)}
print("P1-P40理想最终浓度和实际最终浓度的差值是:", deviations)
print("P1-P40实际浓度与理想浓度的误差比值是:", Error_Ratio)

x_values = [f'P{i}' for i in range(1, 41)]

# 初始化 pm 列表
pm = [math.log(2**(i+1)) for i in range(39)]

Picture.fit_lnk_lnp(pm, k_optimized)

# 绘制理想稳态浓度曲线
Picture.plot_concentrations(x_values, concentrations, final_concentrations = sol[-1, 1:])

# 绘制各个物质的浓度变化曲线
Picture.plot_concentration_curves(t, sol)

# 优化k值后P1-P40实际浓度与理想浓度的误差比值
Picture.plot_error_ratio(x_values, Error)

# 调用动画函数
save_path = r"C:\Users\柴文彬\Desktop\化学动力学\多进程P8\多进程_P8_way1\concentration_animation.gif"
Picture.animate_concentration_curves(t, sol, num_substances=40, save_path=save_path)