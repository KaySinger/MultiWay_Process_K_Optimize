import numpy as np
import math
import matplotlib
matplotlib.use('TkAgg')
from scipy.optimize import minimize, curve_fit
from scipy.integrate import odeint
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# 正态分布模拟，得到的结果用于物质稳态浓度
def simulate_normal_distribution(mu, sigma, total_concentration, scale_factor):
    x_values = np.arange(1, 41)
    concentrations = np.exp(-0.5 * ((x_values - mu) / sigma) ** 2) / (sigma * np.sqrt(2 * np.pi))
    concentrations /= sum(concentrations)
    concentrations *= scale_factor
    return concentrations, x_values

# 初始化 k 和 k_inv 数组
def initialize_k_values(concentrations):
    k = np.zeros(77)
    k_inv = np.zeros(76)
    # P1参与后续反应的初始值猜测
    k[0] = 4
    for i in range(1, 40):
        k[i] = 1 + 0.5 * i
    k_inv[0] = (k[1] * concentrations[0] ** 2) / concentrations[1]
    for i in range(1, 39):
        k_inv[i] = k[i + 1] * concentrations[0] * concentrations[i] / concentrations[i + 1]
    # P2参与后续反应的初始值猜测
    for i in range(37):
        k[i+40] = 1 + 0.2 * i
    k_inv[39] = (k[40] * concentrations[1]**2) / concentrations[3]
    for i in range(36):
        k_inv[i+40] = k[i+41] * concentrations[1] * concentrations[i+2] / concentrations[i+4]
    return list(k) + list(k_inv)

# 定义微分方程进程1
def equations_process1(p, t, k_values):
    k = k_values[:40]
    k_inv = k_values[77:116]
    dpdt = [0] * 41
    dpdt[0] = - k[0] * p[0]
    dpdt[1] = k[0] * p[0] - k[1] * p[1] ** 2 + k_inv[0] * p[2]
    for i in range(1, 39):
        dpdt[1] += k_inv[i] * p[i + 2] - k[i + 1] * p[1] * p[i + 1]
    dpdt[2] = k[1] * p[1] ** 2 + k_inv[1] * p[3] - k_inv[0] * p[2] - k[2] * p[1] * p[2]
    for i in range(3, 40):
        dpdt[i] = 2 * k[i - 1] * p[1] * p[i - 1] + k_inv[i - 1] * p[i + 1] - 2 * k_inv[i - 2] * p[i] - k[i] * p[1] * p[i]
    dpdt[40] = 2 * k[39] * p[1] * p[39] - 2 * k_inv[38] * p[40]
    return dpdt

# 定义微分方程进程2
def equations_process2(p, t, k_values):
    k = k_values[40:77]
    k_inv = k_values[116:]
    dpdt = [0] * 41
    dpdt[2] = k_inv[0] * p[4] - k[0] * p[2]**2
    for i in range(1, 37):
        dpdt[2] += k_inv[i] * p[i+4] - k[i] * p[2] * p[i+2]
    dpdt[3] = k_inv[1] * p[5] - k[1] * p[2] * p[3]
    dpdt[4] = k[0] * p[2]**2 - k_inv[0] * p[4] + k_inv[2] * p[6] - k[2] * p[2] * p[4]
    for i in range(5, 39):
        dpdt[i] = 2 * k[i-4] * p[2] * p[i-2] + k_inv[i-2] * p[i+2] - 2 * k_inv[i-4] * p[i] - k[i-2] * p[2] * p[i]
    dpdt[39] = 2 * k[35] * p[2] * p[37] - 2 * k_inv[35] * p[39]
    dpdt[40] = 2 * k[36] * p[2] * p[38] - 2 * k_inv[36] * p[40]
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
    t = np.linspace(0, 100, 1000)
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
    plt.figure(figsize=(15, 10))
    plt.plot(t, sol[:, 0], label='p0')
    for i in range(1, 11):
        plt.plot(t, sol[:, i], label=f'p{i}')
    plt.legend()
    plt.xlabel('Time')
    plt.ylabel('Concentration')
    plt.title('P0-P10 Concentration over Time')
    plt.grid(True)
    plt.show()

    plt.figure(figsize=(15, 10))
    for i in range(11, 21):
        plt.plot(t, sol[:, i], label=f'p{i}')
    plt.legend()
    plt.xlabel('Time')
    plt.ylabel('Concentration')
    plt.title('P11-P20 Concentration over Time')
    plt.grid(True)
    plt.show()

    plt.figure(figsize=(15, 10))
    for i in range(21, 31):
        plt.plot(t, sol[:, i], label=f'p{i}')
    plt.legend()
    plt.xlabel('Time')
    plt.ylabel('Concentration')
    plt.title('P21-P30 Concentration over Time')
    plt.grid(True)
    plt.show()

    plt.figure(figsize=(15, 10))
    for i in range(31, 41):
        plt.plot(t, sol[:, i], label=f'p{i}')
    plt.legend()
    plt.xlabel('Time')
    plt.ylabel('Concentration')
    plt.title('P31-P40 Concentration over Time')
    plt.grid(True)
    plt.show()

# 假设 t 和 sol 已经计算得到
# 动态展示 P0 - P40 浓度曲线的动画函数
def animate_concentration_curves(t, sol, num_substances=40, interval=1000, save_path = None):
    fig, ax = plt.subplots(figsize=(15, 10))
    lines = [ax.plot([], [], label=f'p{i}')[0] for i in range(num_substances)]

    ax.set_xlim(t[0], t[-1])
    ax.set_ylim(np.min(sol), np.max(sol))
    ax.set_xlabel('Time')
    ax.set_ylabel('Concentration')
    ax.set_title('Concentration over Time')
    ax.legend()
    ax.grid(True)

    def init():
        for line in lines:
            line.set_data([], [])
        return lines

    def update(frame):
        if frame < num_substances:
            for line in lines:
                line.set_data([], [])
            lines[frame].set_data(t, sol[:, frame])
        else:
            for i, line in enumerate(lines):
                line.set_data(t, sol[:, i])
        return lines

    ani = animation.FuncAnimation(fig, update, frames=num_substances + 1, init_func=init, blit=True, repeat=False,
                                  interval=interval)
    ani.save(save_path, writer='pillow', fps=1000 // interval)
    plt.show()

def fit_lnk_lnp(pm, k_optimized):
    # 分别拟合两项公式数据
    popt1, _ = curve_fit(model, pm[:39], np.log(k_optimized[1:40]), maxfev=1000)
    popt2, _ = curve_fit(model, pm[:37], np.log(k_optimized[40:77]), maxfev=1000)

    # 拟合得到的参数
    a1, x1 = popt1
    a2, x2 = popt2
    print(f"原始公式拟合参数: a = {a1}, x = {x1}")
    print(f"P1参与后续反应公式拟合参数: a = {a2}, x = {x2}")

    # 使用拟合参数绘制拟合曲线
    P_fit1 = np.linspace(min(pm[:39]), max(pm[:39]), 100)
    P_fit2 = np.linspace(min(pm[:36]), max(pm[:36]), 100)
    k_fit1 = model(P_fit1, *popt1)
    k_fit2 = model(P_fit2, *popt2)

    # 创建子图
    fig, axs = plt.subplots(2, 1, figsize=(15, 10))

    # 绘制前半部分拟合
    axs[0].scatter(pm[:39], np.log(k_optimized[1:40]), label='Natural data')
    axs[0].plot(P_fit1, k_fit1, color='red', label=f'ln(k) = {a1:.2f} * ln(2^n)^{x1:.2f}')
    axs[0].set_xlabel('polymer')
    axs[0].set_ylabel('ln(k)')
    axs[0].legend()
    axs[0].set_title('oringinal curve fitting')
    axs[0].grid(True)

    # 绘制后半部分拟合
    axs[1].scatter(pm[:37], np.log(k_optimized[40:77]), label='Natural data')
    axs[1].plot(P_fit2, k_fit2, color='blue', label=f'ln(k) = {a2:.2f} * ln(2^n)^{x2:.2f}')
    axs[1].set_xlabel('polymer')
    axs[1].set_ylabel('ln(k)')
    axs[1].legend()
    axs[1].set_title('P1 curve fitting')
    axs[1].grid(True)

    # 调整子图布局
    plt.tight_layout()
    plt.show()

# 定义lnk = a * lnp ^ x的模型
def model(P, a, x):
    return a * P**x


# 模拟正态分布
mu = 20.5
sigma = 10
scale_factor = 10
concentrations, x_values = simulate_normal_distribution(mu, sigma, total_concentration=1.0, scale_factor=scale_factor)
print("理想稳态浓度分布", {f'P{i}': c for i, c in enumerate(concentrations, start=1)})

# 初始K值猜测
initial_guess = initialize_k_values(concentrations)

# 添加参数约束，确保所有k值都是非负的
bounds = [(0, None)] * 77 + [(0, None)] * 76  # 确保长度为 13

# 记录目标函数值
objective_values = []

# 第一次优化
result = minimize(objective, initial_guess, method='L-BFGS-B', bounds=bounds, callback=callback)
k_optimized = result.x
final_precision = result.fun
print(f"优化的最终精度是{final_precision}")

# 输出线程1优化结果
k_result = {f"k{i}": c for i, c in enumerate(k_optimized[:40], start=0)}
k_inv_result = {f"k{i}_inv": c for i, c in enumerate(k_optimized[77:116], start=1)}
print("进程1反应式的k:", k_result)
print("进程1反应式的k_inv:", k_inv_result)

# 输出线程2优化结果
k_result = {f"k{i}": c for i, c in enumerate(k_optimized[40:77], start=0)}
k_inv_result = {f"k{i}_inv": c for i, c in enumerate(k_optimized[116:], start=0)}
print("进程2反应式的k:", k_result)
print("进程2反应式的k_inv:", k_inv_result)

# 利用优化后的参数进行模拟
initial_conditions = [10] + [0] * 40
t = np.linspace(0, 100, 1000)
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
Error_Ratio = {f'Error Ratio P{i}': c for i, c in enumerate(Error, start=1)}
print("P1-P40理想最终浓度和实际最终浓度的差值是:", deviations)
print("P1-P40实际浓度与理想浓度的误差比值是:", Error_Ratio)

x_values = [f'P{i}' for i in range(1, 41)]

# 初始化 pm 列表
pm = [math.log(2**(i+1)) for i in range(39)]

fit_lnk_lnp(pm, k_optimized)

# 绘制理想稳态浓度曲线
plt.figure(figsize=(15, 8))
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

# 优化k值后P1-P70实际浓度与理想浓度的误差比值
plt.figure(figsize=(10, 6))
plt.xlabel("P-Species")
plt.ylabel("P-Error-Ratio")
plt.title("Error Ratio of Concentrations between Ideal and Actual")
plt.xticks(range(len(x_values)), x_values, rotation=90)
plt.plot(range(len(x_values)), Error, label = 'Error-Ratio', marker='o', linestyle='-', color='blue')
plt.grid(True)
plt.show()

# 调用动画函数
save_path = r"C:\Users\柴文彬\Desktop\化学动力学\多进程_P2_way1\concentration_animation.gif"
animate_concentration_curves(t, sol, num_substances=40, save_path=save_path)