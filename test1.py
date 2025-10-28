import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from scipy.stats import norm

# !! 顶会风格的核心：设置全局样式 !!
sns.set_style("whitegrid") # 经典白底网格，清晰易读
plt.rcParams['font.family'] = ['DejaVu Sans'] # 1. 字体：使用学术通用的无衬线字体，如Arial。这里用DejaVu Sans（跨平台）
# 如果系统中安装了Arial，可改为：plt.rcParams['font.family'] = ['Arial']
plt.rcParams['savefig.dpi'] = 300 # 2. 分辨率：保存高清图片
plt.rcParams['savefig.bbox'] = 'tight' # 3. 保存时自动裁剪多余白边
plt.rcParams['font.size'] = 10 # 4. 统一字体大小

# 定义颜色（顶会常用柔和、专业的配色，避免高饱和原色）
male_color = '#1f77b4' # 沉稳的蓝色
female_color = '#ff7f0e' # 温暖的橙色

# 定义参数（使用你图片中的数据）
male_height_mean = 169.7
male_height_std = 6
female_height_mean = 158.0
female_height_std = 5

# 生成数据点
x = np.linspace(140, 190, 500)
male_pdf = norm.pdf(x, male_height_mean, male_height_std)
female_pdf = norm.pdf(x, female_height_mean, female_height_std)

# 创建图形和坐标轴（这是更专业的写法）
fig, ax = plt.subplots(figsize=(8, 5)) # 宽高比更协调

# 使用Seaborn的线图功能绘制，线条更平滑
sns.lineplot(x=x, y=male_pdf, color=male_color, linewidth=2, label=f'Male ($μ$={male_height_mean}, $σ$={male_height_std})', ax=ax)
sns.lineplot(x=x, y=female_pdf, color=female_color, linewidth=2, label=f'Female ($μ$={female_height_mean}, $σ$={female_height_std})', ax=ax)

# !! 精细化修饰 !!
ax.set_xlabel('Height (cm)', fontsize=12) # 标签用英文是顶会惯例
ax.set_ylabel('Probability Density', fontsize=12)
ax.set_title('Probability Density Function of Adult Height by Gender in China', fontsize=13, pad=15) # pad让标题不挤

# 设置坐标轴刻度更密集、更清晰
ax.tick_params(axis='both', which='major', labelsize=10)
ax.set_xlim(140, 190)

# 添加图例，并放在最佳位置
ax.legend(frameon=True, fancybox=False, edgecolor='black', fontsize=10) # 去掉fancybox，加边框，更专业

# 最后，保存为多种格式，特别是PDF（矢量图）
plt.savefig('height_pdf_top_quality.png') # 用于PPT等
plt.savefig('height_pdf_top_quality.pdf') # 用于论文排版，无限清晰
plt.show()