import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from scipy.stats import gaussian_kde, multivariate_normal
from mpl_toolkits.mplot3d import Axes3D

# 设置顶会风格
sns.set_style("whitegrid")
plt.rcParams['font.family'] = ['DejaVu Sans']
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['savefig.bbox'] = 'tight'
plt.rcParams['font.size'] = 10

# 模拟外貌数据（这里以面部特征为例）
np.random.seed(42)

# 生成模拟数据：脸型比例、五官协调度、对称性等维度
n_samples = 1000

# 三个主要维度：脸型协调度、五官立体度、整体对称性
data = {
    'facial_harmony': np.random.normal(0.7, 0.15, n_samples),  # 脸型协调度
    'facial_symmetry': np.random.normal(0.8, 0.1, n_samples),  # 面部对称性
    'feature_distinctness': np.random.normal(0.6, 0.2, n_samples)  # 五官立体度
}

df = pd.DataFrame(data)

# 方法1：单变量分布图
fig, axes = plt.subplots(2, 2, figsize=(12, 10))

# 脸型协调度分布
sns.histplot(df['facial_harmony'], kde=True, ax=axes[0,0], color='#1f77b4', alpha=0.7)
axes[0,0].set_title('Facial Harmony Distribution')
axes[0,0].set_xlabel('Harmony Score')
axes[0,0].set_ylabel('Density')

# 面部对称性分布
sns.histplot(df['facial_symmetry'], kde=True, ax=axes[0,1], color='#ff7f0e', alpha=0.7)
axes[0,1].set_title('Facial Symmetry Distribution')
axes[0,1].set_xlabel('Symmetry Score')
axes[0,1].set_ylabel('Density')

# 五官立体度分布
sns.histplot(df['feature_distinctness'], kde=True, ax=axes[1,0], color='#2ca02c', alpha=0.7)
axes[1,0].set_title('Feature Distinctness Distribution')
axes[1,0].set_xlabel('Distinctness Score')
axes[1,0].set_ylabel('Density')

# 综合评分分布（三个维度的加权平均）
df['composite_score'] = (df['facial_harmony'] * 0.4 + 
                        df['facial_symmetry'] * 0.3 + 
                        df['feature_distinctness'] * 0.3)

sns.histplot(df['composite_score'], kde=True, ax=axes[1,1], color='#d62728', alpha=0.7)
axes[1,1].set_title('Composite Attractiveness Score')
axes[1,1].set_xlabel('Composite Score')
axes[1,1].set_ylabel('Density')

plt.tight_layout()
plt.savefig('facial_features_univariate_distributions.png')
plt.show()