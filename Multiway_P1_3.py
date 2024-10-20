import numpy as np
from scipy.optimize import minimize
from scipy.integrate import odeint
import matplotlib.pyplot as plt

# 正态分布模拟，得到的结果用于物质稳态浓度
def simulate_normal_distribution(mu, sigma, total_concentration, scale_factor):
    x_values = np.arange(1, 41)
    concentrations = np.exp(-0.5 * ((x_values - mu) / sigma) ** 2) / (sigma * np.sqrt(2 * np.pi))
    concentrations /= sum(concentrations)
    concentrations *= scale_factor
    return concentrations, x_values

# 初始化 k 和 k_inv 数组
def initialize_k_values(concentrations):
    k = np.zeros(78)
    k_inv = np.zeros(77)
    k[0], k[1], k[2] = 2, 1, 2
    # P0不参与后续反应的初始值猜测
    k_inv[0] = (k[1] * concentrations[0] ** 2) / concentrations[1]
    k_inv[1] = (k[2] * concentrations[1] ** 2) / concentrations[2]
    for i in range(3, 40):
        k[i] = k[i-1] * concentrations[i-2]**2 / (concentrations[i-1] ** 2)
        k_inv[i-1] = k_inv[i-2] * concentrations[i-1] / concentrations[i]
    # P0参与后续反应的初始值猜测
    k[40] = 0.1
    k_inv[39] = k[40] * concentrations[0] * concentrations[1] / concentrations[2]
    for i in range(41, 78):
        k[i] = k[i-1] * concentrations[0] * concentrations[i-40] / (concentrations[0] * concentrations[i-39])
        k_inv[i-1] = k_inv[i-2] * concentrations[i-39] / concentrations[i-38]
    return list(k) + list(k_inv)

# 定义微分方程进程1
def equations_process1(p, t, k_values):
    k = k_values[:40]
    k_inv = k_values[78:117]
    dpdt = [0] * 41
    dpdt[0] = - k[0] * p[0]
    dpdt[1] = k[0] * p[0] - k[1] * p[1]**2 + k_inv[0] * p[2]
    dpdt[2] = k[1] * p[1]**2 + k_inv[1] * p[3] - k_inv[0] * p[2] - k[2] * p[2]**2
    for i in range(3, 40):
        dpdt[i] = k[i-1] * p[i-1]**2 + k_inv[i-1] * p[i+1] - k_inv[i-2] * p[i] - k[i] * p[i]**2
    dpdt[40] = k[39] * p[39]**2 - k_inv[38] * p[40]
    return dpdt

# 定义微分方程进程2
def equations_process2(p, t, k_values):
    k = k_values[40:78]
    k_inv = k_values[117:155]
    dpdt = [0] * 41
    for i in range(1, 39):
        dpdt[1] += k_inv[i-1] * p[i+2] - k[i-1] * p[1] * p[i+1]
    dpdt[2] = k_inv[0] * p[3] - k[0] * p[1] * p[2]
    for i in range(3, 40):
        dpdt[i] = 2 * k[i-3] * p[1] * p[i-1] + k_inv[i-2] * p[i+1] - 2 * k_inv[i-3] * p[i] - k[i-2] * p[1] * p[i]
    dpdt[40] = 2 * k[37] * p[1] * p[39] - 2 * k_inv[37] * p[40]
    return dpdt

# 定义总的微分方程
def equations(p, t, k_values):
    dpdt_process1 = equations_process1(p, t, k_values)
    dpdt_process2 = equations_process2(p, t, k_values)
    dpdt = [dpdt_process1[i] + dpdt_process2[i] for i in range(41)]
    return dpdt

# 定义目标函数
def objective(k):
    initial_conditions = [10] + [0] * 40
    t = np.linspace(0, 5000, 5000)
    sol = odeint(equations, initial_conditions, t, args=(k,))
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

# 绘图函数
def plot_concentration_curves(t, sol):
    plt.figure(figsize=(20, 10))
    plt.plot(t, sol[:, 0], label='p0')
    for i in range(1, 6):
        plt.plot(t, sol[:, i], label=f'p{i}')
    plt.legend()
    plt.xlabel('Time')
    plt.ylabel('Concentration')
    plt.title('P1-P5 Concentration over Time')
    plt.grid(True)
    plt.show()

    plt.figure(figsize=(20, 10))
    for i in range(6, 11):
        plt.plot(t, sol[:, i], label=f'p{i}')
    plt.legend()
    plt.xlabel('Time')
    plt.ylabel('Concentration')
    plt.title('P6-P10 Concentration over Time')
    plt.grid(True)
    plt.show()

    plt.figure(figsize=(20, 10))
    for i in range(11, 21):
        plt.plot(t, sol[:, i], label=f'p{i}')
    plt.legend()
    plt.xlabel('Time')
    plt.ylabel('Concentration')
    plt.title('P11-P20 Concentration over Time')
    plt.grid(True)
    plt.show()

    plt.figure(figsize=(20, 10))
    for i in range(21, 31):
        plt.plot(t, sol[:, i], label=f'p{i}')
    plt.legend()
    plt.xlabel('Time')
    plt.ylabel('Concentration')
    plt.title('P21-P30 Concentration over Time')
    plt.grid(True)
    plt.show()

    plt.figure(figsize=(20, 10))
    for i in range(31, 41):
        plt.plot(t, sol[:, i], label=f'p{i}')
    plt.legend()
    plt.xlabel('Time')
    plt.ylabel('Concentration')
    plt.title('P31-P40 Concentration over Time')
    plt.grid(True)
    plt.show()



# 模拟正态分布
mu = 20.5
sigma = 10
scale_factor = 10
concentrations, x_values = simulate_normal_distribution(mu, sigma, total_concentration=1.0, scale_factor=scale_factor)
print("理想稳态浓度分布", {f'P{i}': c for i, c in enumerate(concentrations, start=1)})

# 初始K值猜测
initial_guess = initialize_k_values(concentrations)

# 添加参数约束，确保所有k值都是非负的
bounds = [(0, 3)] * 78 + [(0, 1)] * 77  # 确保长度为 13

# 记录目标函数值
objective_values = []

# 第一次优化
result = minimize(objective, initial_guess, method='L-BFGS-B', bounds=bounds, callback=callback)
k_optimized = result.x
final_precision = result.fun
print(f"优化的最终精度是{final_precision}")

# 输出进程1优化结果
k_process1_result = {f"k{i}": c for i, c in enumerate(k_optimized[:40], start=0)}
k_inv_process1_result = {f"k{i}_inv": c for i, c in enumerate(k_optimized[78:117], start=1)}
print("优化后的k", k_process1_result)
print("k_inv", k_inv_process1_result)

# 输出进程2优化结果
k_process2_result = {f"k{i}": c for i, c in enumerate(k_optimized[40:78], start=0)}
k_inv_process2_result = {f"k{i}_inv": c for i, c in enumerate(k_optimized[117:], start=0)}
print("优化后的k", k_process2_result)
print("k_inv", k_inv_process2_result)

# 利用优化后的参数进行模拟
initial_conditions = [10] + [0] * 40
t = np.linspace(0, 5000, 5000)
sol = odeint(equations, initial_conditions, t, args=(k_optimized,))

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
Error_Ratio = {f'Error Ratio of P{i}': c for i, c in enumerate(Error, start=1)}
print("P1-P5理想最终浓度和实际最终浓度的差值是", deviations)
print("P1-P5实际浓度与理想浓度的误差比值是", Error_Ratio)

x_values = [f'P{i}' for i in range(1, 41)]

# 绘制理想稳态浓度曲线
plt.figure(figsize=(20, 10))
plt.xlabel("P-Species")
plt.ylabel("P-Concentrations")
plt.title("Ideal Concentrations and Actual Concentrations")
plt.xticks(range(len(x_values)), x_values, rotation=90)
final_concentrations = sol[-1, 1:]
plt.plot(range(len(x_values)), concentrations, label = 'Ideal Concentrations', marker='o', linestyle='-', color='blue')
plt.plot(range(len(x_values)), final_concentrations, label = 'Actual Concentrations', marker='o', linestyle='-', color='red')
plt.grid(True)
plt.show()

# 绘制各个物质的浓度变化曲线
plot_concentration_curves(t, sol)

# 优化k值后P1-P5实际浓度与理想浓度的误差比值
plt.figure(figsize=(10, 5))
plt.xlabel("P-Species")
plt.ylabel("P-Error-Ratio")
plt.title("Error Ratio of Concentrations between Ideal and Actual")
plt.xticks(range(len(x_values)), x_values, rotation=90)
plt.plot(range(len(x_values)), Error, label = 'Error-Ratio', marker='o', linestyle='-', color='blue')
plt.grid(True)
plt.show()