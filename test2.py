import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from scipy.stats import gaussian_kde

# !! 顶会风格的核心：设置全局样式 !!
sns.set_style("whitegrid") # 经典白底网格，清晰易读
plt.rcParams['font.family'] = ['DejaVu Sans'] # 学术通用的无衬线字体
plt.rcParams['savefig.dpi'] = 300 # 保存高清图片
plt.rcParams['savefig.bbox'] = 'tight' # 自动裁剪多余白边
plt.rcParams['font.size'] = 10 # 统一字体大小

# 定义学历分布数据
education_data = {
    'Education Level': ['No School', 'Primary', 'Junior High', 'Senior High', 
                       'Associate', 'Bachelor', 'Master+'],
    'Percentage': [3.92, 24.9, 35.0, 15.7, 9.5, 8.5, 0.92]
}

df = pd.DataFrame(education_data)

# 为每个学历层次分配数值并生成模拟数据点
education_values = [0, 1, 2, 3, 4, 5, 6]  # 数值映射
np.random.seed(42)  # 确保可重复性

# 根据百分比生成模拟数据
simulated_data = []
for i, percentage in enumerate(df['Percentage']):
    # 为每个学历层次生成相应数量的数据点
    num_points = int(percentage * 100)  # 放大倍数以获得足够数据点
    # 在每个学历值周围添加随机变化
    simulated_data.extend(np.random.normal(loc=education_values[i], scale=0.2, size=num_points))

simulated_data = np.array(simulated_data)

# 创建图形和坐标轴
fig, ax = plt.subplots(figsize=(10, 6))

# 计算核密度估计
kde = gaussian_kde(simulated_data, bw_method=0.3)  # 调整带宽控制平滑度
x_range = np.linspace(-1, 7, 1000)
density = kde(x_range)

# 使用Seaborn风格绘制平滑曲线
sns.lineplot(x=x_range, y=density, color='#1f77b4', linewidth=2.5, ax=ax, 
             label='Education Distribution KDE')

# 添加原始数据点的分布（透明度较低）
ax.hist(simulated_data, bins=50, density=True, alpha=0.3, color='#2ca02c', 
        edgecolor='black', linewidth=0.5, label='Histogram')

# !! 精细化修饰 !!
ax.set_xlabel('Education Level', fontsize=12)
ax.set_ylabel('Probability Density', fontsize=12)
ax.set_title('Probability Density of Education Distribution in China', fontsize=13, pad=15)

# 设置x轴刻度和标签
ax.set_xticks(education_values)
ax.set_xticklabels(df['Education Level'], rotation=45, ha='right')

# 设置坐标轴范围
ax.set_xlim(-0.5, 6.5)

# 添加网格和图例
ax.grid(True, linestyle='--', alpha=0.7)
ax.legend(frameon=True, fancybox=False, edgecolor='black', fontsize=10, 
          loc='upper left')

# 添加统计信息文本框
stats_text = f'Total Sample: {len(simulated_data):,}\nBandwidth: {kde.factor:.3f}'
ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, verticalalignment='top',
        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8), fontsize=9)

# 保存为多种格式
plt.savefig('education_distribution_top_quality.png')
plt.savefig('education_distribution_top_quality.pdf')  # 矢量图用于论文

plt.tight_layout()
plt.show()

# 可选：创建对比图显示不同带宽的效果
fig, axes = plt.subplots(2, 2, figsize=(12, 8))
bw_methods = [0.1, 0.3, 0.5, 0.8]  # 不同带宽参数

for i, bw in enumerate(bw_methods):
    ax = axes[i//2, i%2]
    kde_temp = gaussian_kde(simulated_data, bw_method=bw)
    density_temp = kde_temp(x_range)
    
    sns.lineplot(x=x_range, y=density_temp, color='#1f77b4', linewidth=2, ax=ax)
    ax.hist(simulated_data, bins=50, density=True, alpha=0.3, color='#2ca02c')
    ax.set_title(f'Bandwidth = {bw}')
    ax.set_xlabel('Education Level')
    ax.set_ylabel('Density')
    ax.set_xticks(education_values)
    ax.set_xticklabels(df['Education Level'], rotation=45, ha='right')

plt.tight_layout()
plt.savefig('education_kde_bandwidth_comparison.png', dpi=300)
plt.show()