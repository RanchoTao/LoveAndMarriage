import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

# 设置中文字体，防止乱码
plt.rcParams['font.sans-serif'] = ['SimHei']  # 例如使用黑体
plt.rcParams['axes.unicode_minus'] = False    # 正确显示负号

# 定义男女体重的分布参数（使用您提供的参数）
male_mean = 70
male_std = 10
female_mean = 55
female_std = 8

# 生成一个在特定范围内均匀分布的x值数组，用于绘制平滑曲线
x = np.linspace(20, 110, 1000)  # 体重范围从20kg到110kg

# 计算男性和女性在该x值处的概率密度函数（PDF）值
male_pdf = norm.pdf(x, male_mean, male_std)
female_pdf = norm.pdf(x, female_mean, female_std)

# 开始绘图
plt.figure(figsize=(10, 6))  # 设置图片大小

# 绘制男性体重PDF曲线
plt.plot(x, male_pdf, color='blue', linewidth=2, label=f'男性 (μ={male_mean}, σ={male_std})')
# 绘制女性体重PDF曲线
plt.plot(x, female_pdf, color='red', linewidth=2, label=f'女性 (μ={female_mean}, σ={female_std})')

# 添加图形标识
plt.title('中国成年男女体重分布概率密度函数（PDF）', fontsize=14)
plt.xlabel('体重 (kg)', fontsize=12)
plt.ylabel('概率密度', fontsize=12)
plt.legend(fontsize=11)  # 显示图例
plt.grid(True, linestyle='--', alpha=0.7)  # 添加网格线，便于读数

# 显示图形
plt.tight_layout()
plt.show()