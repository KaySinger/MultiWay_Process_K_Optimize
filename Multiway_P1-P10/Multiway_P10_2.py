import numpy as np
import math, random
import matplotlib
matplotlib.use('TkAgg')
from scipy.optimize import minimize
from scipy.integrate import odeint
from Diffusions import Diffusion_Origin
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
    k = np.zeros(339)
    k_inv = np.zeros(338)
    k[0] = 3
    # P1不参与后续反应的初始值猜测
    for i in range(1, 40):
        k[i] = 1.5 + random.uniform(0.5, 1) * i
    k_inv[0] = (k[1] * concentrations[0] ** 2) / concentrations[1]
    for i in range(1, 39):
        k_inv[i] = (k[i + 1] * concentrations[i] ** 2) / concentrations[i + 1]
    # P1参与后续反应的初始值猜测
    for i in range(38):
        k[i + 40] = 1.2 + random.uniform(0.3, 0.5) * i
    for i in range(38):
        k_inv[i + 39] = k[i + 40] * concentrations[0] * concentrations[i + 1] / concentrations[i + 2]
    # P2参与后续反应的初始值猜测
    for i in range(37):
        k[i + 78] = 1 + random.uniform(0.3, 0.5) * i
    k_inv[77] = (k[78] * concentrations[1] ** 2) / concentrations[3]
    for i in range(1, 37):
        k_inv[i + 77] = k[i + 78] * concentrations[1] * concentrations[i + 1] / concentrations[i + 3]
    # P3参与后续反应的初始值猜测
    for i in range(35):
        k[i + 115] = 1 + random.uniform(0.3, 0.5) * i
    k_inv[114] = (k[115] * concentrations[2] ** 2) / concentrations[5]
    for i in range(34):
        k_inv[i + 115] = k[i + 116] * concentrations[2] * concentrations[i + 3] / concentrations[i + 6]
    # P4参与后续反应的初始值猜测
    for i in range(33):
        k[i+150] = 1 + random.uniform(0.3, 0.5) * i
    k_inv[149] = (k[150] * concentrations[3]**2) / concentrations[7]
    for i in range(32):
        k_inv[i+150] = k[i+151] * concentrations[3] * concentrations[i+4] / concentrations[i+8]
    # P5参与后续反应的初始值猜测
    for i in range(31):
        k[i + 183] = 1 + random.uniform(0.3, 0.5) * i
    k_inv[182] = (k[183] * concentrations[4] ** 2) / concentrations[9]
    for i in range(30):
        k_inv[i + 183] = k[i + 184] * concentrations[4] * concentrations[i + 5] / concentrations[i + 10]
    # P6参与后续反应的初始值猜测
    for i in range(29):
        k[i + 214] = 1 + random.uniform(0.3, 0.5) * i
    k_inv[213] = (k[214] * concentrations[5] ** 2) / concentrations[11]
    for i in range(28):
        k_inv[i + 214] = k[i + 215] * concentrations[5] * concentrations[i + 6] / concentrations[i + 12]
    # P7参与后续反应的初始值猜测
    for i in range(27):
        k[i + 243] = 1 + random.uniform(0.3, 0.5) * i
    k_inv[242] = (k[243] * concentrations[6] ** 2) / concentrations[13]
    for i in range(26):
        k_inv[i + 243] = k[i + 244] * concentrations[6] * concentrations[i + 7] / concentrations[i + 14]
    # P8参与后续反应的初始值猜测
    for i in range(25):
        k[i + 270] = 1 + random.uniform(0.3, 0.5) * i
    k_inv[269] = (k[270] * concentrations[7] ** 2) / concentrations[15]
    for i in range(24):
        k_inv[i + 270] = k[i + 271] * concentrations[7] * concentrations[i + 8] / concentrations[i + 16]
    # P9参与后续反应的初始值猜测
    for i in range(23):
        k[i + 295] = 1 + random.uniform(0.3, 0.5) * i
    k_inv[294] = (k[295] * concentrations[8] ** 2) / concentrations[17]
    for i in range(22):
        k_inv[i + 295] = k[i + 296] * concentrations[8] * concentrations[i + 9] / concentrations[i + 18]
    # P10参与后续反应的初始值猜测
    for i in range(21):
        k[i + 318] = 1 + random.uniform(0.3, 0.5) * i
    k_inv[317] = (k[318] * concentrations[9] ** 2) / concentrations[19]
    for i in range(20):
        k_inv[i + 318] = k[i + 319] * concentrations[9] * concentrations[i + 10] / concentrations[i + 20]
    return list(k), list(k_inv)

# 定义总的微分方程
def equations(p, t, k, k_inv):
    dpdt_process1 = Diffusion_Origin.equations_process1(p, t, k, k_inv)
    dpdt_process2 = Diffusion_Origin.equations_process2(p, t, k, k_inv)
    dpdt_process3 = Diffusion_Origin.equations_process3(p, t, k, k_inv)
    dpdt_process4 = Diffusion_Origin.equations_process4(p, t, k, k_inv)
    dpdt_process5 = Diffusion_Origin.equations_process5(p, t, k, k_inv)
    dpdt_process6 = Diffusion_Origin.equations_process6(p, t, k, k_inv)
    dpdt_process7 = Diffusion_Origin.equations_process7(p, t, k, k_inv)
    dpdt_process8 = Diffusion_Origin.equations_process8(p, t, k, k_inv)
    dpdt_process9 = Diffusion_Origin.equations_process9(p, t, k, k_inv)
    dpdt_process10 = Diffusion_Origin.equations_process10(p, t, k, k_inv)
    dpdt_process11 = Diffusion_Origin.equations_process11(p, t, k, k_inv)
    dpdt = [dpdt_process1[i] + dpdt_process2[i] + dpdt_process3[i] + dpdt_process4[i] + dpdt_process5[i] + dpdt_process6[i] + dpdt_process7[i] + dpdt_process8[i]
            + dpdt_process9[i] + dpdt_process10[i] + dpdt_process11[i] for i in range(41)]
    return dpdt

# 定义目标函数
def objective(params):
    k = params[:339]
    k_inv = params[339:]
    initial_conditions = [10] + [0] * 40
    t = np.linspace(0, 5, 500)
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
bounds = [(0, None)] * len(k_initial) + [(0, None)] * len(k_inv_initial)

# 记录目标函数值
objective_values = []

# 第一次优化
result = minimize(objective, initial_guess, method='L-BFGS-B', bounds=bounds, callback=callback)
k_optimized = result.x[:339]
k_inv_optimized = result.x[339:]
final_precision = result.fun
print(f"优化的最终精度是{final_precision}")

# 输出线程1优化结果
k_result = {f"k{i}": c for i, c in enumerate(k_optimized[:40], start=0)}
k_inv_result = {f"k{i}_inv": c for i, c in enumerate(k_inv_optimized[:39], start=1)}
print("进程1反应式的k:", k_result)
print("进程1反应式的k_inv:", k_inv_result)

# 输出线程2优化结果
k_result = {f"k{i}": c for i, c in enumerate(k_optimized[40:78], start=0)}
k_inv_result = {f"k{i}_inv": c for i, c in enumerate(k_inv_optimized[39:77], start=0)}
print("进程2反应式的k:", k_result)
print("进程2反应式的k_inv:", k_inv_result)

# 输出线程3优化结果
k_result = {f"k{i}": c for i, c in enumerate(k_optimized[78:115], start=0)}
k_inv_result = {f"k{i}_inv": c for i, c in enumerate(k_inv_optimized[77:114], start=0)}
print("进程3反应式的k:", k_result)
print("进程3反应式的k_inv:", k_inv_result)

# 输出线程4优化结果
k_result = {f"k{i}": c for i, c in enumerate(k_optimized[115:150], start=0)}
k_inv_result = {f"k{i}_inv": c for i, c in enumerate(k_inv_optimized[114:149], start=0)}
print("进程4反应式的k:", k_result)
print("进程4反应式的k_inv:", k_inv_result)

# 输出线程5优化结果
k_result = {f"k{i}": c for i, c in enumerate(k_optimized[150:183], start=0)}
k_inv_result = {f"k{i}_inv": c for i, c in enumerate(k_inv_optimized[149:182], start=0)}
print("进程5反应式的k:", k_result)
print("进程5反应式的k_inv:", k_inv_result)

# 输出线程6优化结果
k_result = {f"k{i}": c for i, c in enumerate(k_optimized[183:214], start=0)}
k_inv_result = {f"k{i}_inv": c for i, c in enumerate(k_inv_optimized[182:213], start=0)}
print("进程6反应式的k:", k_result)
print("进程6反应式的k_inv:", k_inv_result)

# 输出线程7优化结果
k_result = {f"k{i}": c for i, c in enumerate(k_optimized[214:243], start=0)}
k_inv_result = {f"k{i}_inv": c for i, c in enumerate(k_inv_optimized[213:242], start=0)}
print("进程7反应式的k:", k_result)
print("进程7反应式的k_inv:", k_inv_result)

# 输出线程8优化结果
k_result = {f"k{i}": c for i, c in enumerate(k_optimized[243:270], start=0)}
k_inv_result = {f"k{i}_inv": c for i, c in enumerate(k_inv_optimized[242:269], start=0)}
print("进程8反应式的k:", k_result)
print("进程8反应式的k_inv:", k_inv_result)

# 输出线程9优化结果
k_result = {f"k{i}": c for i, c in enumerate(k_optimized[270:295], start=0)}
k_inv_result = {f"k{i}_inv": c for i, c in enumerate(k_inv_optimized[269:294], start=0)}
print("进程9反应式的k:", k_result)
print("进程9反应式的k_inv:", k_inv_result)

# 输出线程10优化结果
k_result = {f"k{i}": c for i, c in enumerate(k_optimized[295:318], start=0)}
k_inv_result = {f"k{i}_inv": c for i, c in enumerate(k_inv_optimized[294:317], start=0)}
print("进程10反应式的k:", k_result)
print("进程10反应式的k_inv:", k_inv_result)

# 输出线程11优化结果
k_result = {f"k{i}": c for i, c in enumerate(k_optimized[318:339], start=0)}
k_inv_result = {f"k{i}_inv": c for i, c in enumerate(k_inv_optimized[317:338], start=0)}
print("进程11反应式的k:", k_result)
print("进程11反应式的k_inv:", k_inv_result)

# 利用优化后的参数进行模拟
initial_conditions = [10] + [0] * 40
t = np.linspace(0, 5, 500)
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
save_path = r"C:\Users\柴文彬\Desktop\化学动力学\多进程P10\多进程_P10_way2\concentration_animation.gif"
Picture.animate_concentration_curves(t, sol, num_substances=40, save_path=save_path)