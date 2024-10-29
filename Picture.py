import numpy as np
import math
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
import matplotlib.animation as animation

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

# 拟合曲线函数
def fit_lnk_lnp(pm, k_optimized):
    diffs = np.diff(np.log(k_optimized[1:40]))

    # 找到变化率最大的点作为分界点
    split_index = max(np.argmax(np.abs(diffs)) + 1, 5)

    # 分别拟合前后数据
    popt1, _ = curve_fit(model, pm[:split_index], np.log(k_optimized[1:split_index + 1]), maxfev=1000)
    popt2, _ = curve_fit(model, pm[split_index:], np.log(k_optimized[split_index + 1:40]), maxfev=1000)
    # 整体拟合
    popt_all, _ = curve_fit(model, pm, np.log(k_optimized[1:40]), maxfev=1000)

    # 拟合得到的参数
    a1, x1 = popt1
    a2, x2 = popt2
    a_all, x_all = popt_all
    print(f"前半部分拟合参数: a = {a1}, x = {x1}")
    print(f"后半部分拟合参数: a = {a2}, x = {x2}")
    print(f"整体拟合参数: a = {a_all}, x = {x_all}")

    # 使用拟合参数绘制拟合曲线
    P_fit1 = np.linspace(min(pm[:split_index]), max(pm[:split_index]), 100)
    P_fit2 = np.linspace(min(pm[split_index:]), max(pm[split_index:]), 100)
    P_fit_all = np.linspace(min(pm), max(pm), 100)
    k_fit1 = model(P_fit1, *popt1)
    k_fit2 = model(P_fit2, *popt2)
    k_fit_all = model(P_fit_all, *popt_all)

    # 创建子图
    fig, axs = plt.subplots(4, 1, figsize=(10, 10))

    # 绘制前半部分拟合
    axs[0].scatter(pm[:split_index], np.log(k_optimized[1:split_index + 1]), label='Natural data')
    axs[0].plot(P_fit1, k_fit1, color='red', label=f'ln(k) = {a1:.2f} * ln(2^n)^{x1:.2f}')
    axs[0].set_xlabel('polymer')
    axs[0].set_ylabel('ln(k)')
    axs[0].legend()
    axs[0].set_title('front curve fitting')
    axs[0].grid(True)

    # 绘制后半部分拟合
    axs[1].scatter(pm[split_index:], np.log(k_optimized[split_index + 1:40]), label='Natural data')
    axs[1].plot(P_fit2, k_fit2, color='blue', label=f'ln(k) = {a2:.2f} * ln(2^n)^{x2:.2f}')
    axs[1].set_xlabel('polymer')
    axs[1].set_ylabel('ln(k)')
    axs[1].legend()
    axs[1].set_title('behind curve fitting')
    axs[1].grid(True)

    # 绘制前后加在一起的拟合
    axs[2].scatter(pm, np.log(k_optimized[1:40]), label='Natural data')
    axs[2].plot(P_fit1, k_fit1, color='red', label=f'front: ln(k) = {a1:.2f} * ln(2^n)^{x1:.2f}')
    axs[2].plot(P_fit2, k_fit2, color='blue', label=f'behind: ln(k) = {a2:.2f} * ln(2^n)^{x2:.2f}')
    axs[2].set_xlabel('polymer')
    axs[2].set_ylabel('ln(k)')
    axs[2].legend()
    axs[2].set_title('combined curve fitting')
    axs[2].grid(True)

    # 绘制整体拟合
    axs[3].scatter(pm, np.log(k_optimized[1:40]), label='Natural data')
    axs[3].plot(P_fit_all, k_fit_all, color='green', label=f'all: ln(k) = {a_all:.2f} * ln(2^n)^{x_all:.2f}')
    axs[3].set_xlabel('polymer')
    axs[3].set_ylabel('ln(k)')
    axs[3].legend()
    axs[3].set_title('overall curve fitting')
    axs[3].grid(True)

    # 调整子图布局
    plt.tight_layout()
    plt.show()

# 理想和实际稳态浓度分布曲线
def plot_concentrations(x_values, concentrations, final_concentrations):
    plt.figure(figsize=(15, 8))
    plt.xlabel("P-Species")
    plt.ylabel("P-Concentrations")
    plt.title("Ideal Concentrations and Actual Concentrations")
    plt.xticks(range(len(x_values)), x_values, rotation=90)
    plt.plot(range(len(x_values)), concentrations, label='Ideal Concentrations', marker='o', linestyle='-', color='blue')
    plt.plot(range(len(x_values)), final_concentrations, label='Actual Concentrations', marker='o', linestyle='-', color='red')

    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

# 误差比值图
def plot_error_ratio(x_values, Error):
    plt.figure(figsize=(10, 6))
    plt.xlabel("P-Species")
    plt.ylabel("P-Error-Ratio")
    plt.title("Error Ratio of Concentrations between Ideal and Actual")
    plt.xticks(range(len(x_values)), x_values, rotation=90)

    plt.plot(range(len(x_values)), Error, label='Error-Ratio', marker='o', linestyle='-', color='blue')

    plt.grid(True)
    plt.tight_layout()
    plt.show()

# 定义lnk = a * lnp ^ x的模型
def model(P, a, x):
    return a * P**x

# 初始化 pm 列表
pm = [math.log(2**(i+1)) for i in range(39)]