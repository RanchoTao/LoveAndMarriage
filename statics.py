import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import pandas as pd
from matplotlib.font_manager import FontProperties
import warnings
warnings.filterwarnings('ignore')

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

class DatingMarketVisualizer:
    def __init__(self):
        """初始化，基于中国统计局数据的参数"""
        # 基于中国社会统计数据的分布参数
        self.distribution_params = {
            'age': {'mean': 30, 'std': 8, 'min': 18, 'max': 60, 'type': 'normal'},
            'height_male': {'mean': 172, 'std': 6, 'min': 150, 'max': 200, 'type': 'normal'},
            'height_female': {'mean': 160, 'std': 5, 'min': 140, 'max': 185, 'type': 'normal'},
            'weight_male': {'mean': 70, 'std': 10, 'min': 45, 'max': 120, 'type': 'normal'},
            'weight_female': {'mean': 55, 'std': 8, 'min': 40, 'max': 90, 'type': 'normal'},
            'education': {'mean': 6.5, 'std': 2.5, 'min': 0, 'max': 10, 'type': 'normal'},
            'appearance': {'mean': 60, 'std': 15, 'min': 0, 'max': 100, 'type': 'normal'},
            'iq': {'mean': 100, 'std': 15, 'min': 70, 'max': 140, 'type': 'normal'},
            'assets': {'mean': 2.5, 'std': 1.2, 'min': 0, 'max': 10, 'type': 'lognormal'},  # 对数正态
            'income': {'mean': 15, 'std': 10, 'min': 3, 'max': 100, 'type': 'lognormal'}
        }
        
        # 分类变量的分布（基于统计数据）
        self.categorical_dists = {
            'region': ['一线城市', '二线城市', '三线城市', '农村地区'],
            'region_probs': [0.15, 0.35, 0.30, 0.20],
            'hukou': ['一线户籍', '二线户籍', '三线户籍', '农村户籍'],
            'hukou_probs': [0.10, 0.25, 0.35, 0.30],
            'health': ['优秀', '良好', '一般', '较差'],
            'health_probs': [0.3, 0.4, 0.2, 0.1],
            'family_bg': ['富裕', '中产', '工薪', '困难'],
            'family_bg_probs': [0.1, 0.3, 0.4, 0.2]
        }

    def generate_sample_data(self, n_samples=10000):
        """生成模拟的中国婚恋市场数据"""
        np.random.seed(42)  # 保证可重复性
        
        data = []
        for i in range(n_samples):
            # 性别（男略多）
            gender = np.random.choice(['男', '女'], p=[0.51, 0.49])
            
            # 连续变量
            if gender == '男':
                height = np.random.normal(self.distribution_params['height_male']['mean'], 
                                        self.distribution_params['height_male']['std'])
                weight = np.random.normal(self.distribution_params['weight_male']['mean'], 
                                        self.distribution_params['weight_male']['std'])
            else:
                height = np.random.normal(self.distribution_params['height_female']['mean'], 
                                        self.distribution_params['height_female']['std'])
                weight = np.random.normal(self.distribution_params['weight_female']['mean'], 
                                        self.distribution_params['weight_female']['std'])
            
            # 限制在合理范围内
            height = max(self.distribution_params['height_male']['min'] if gender == '男' else self.distribution_params['height_female']['min'],
                        min(self.distribution_params['height_male']['max'] if gender == '男' else self.distribution_params['height_female']['max'], height))
            weight = max(self.distribution_params['weight_male']['min'] if gender == '男' else self.distribution_params['weight_female']['min'],
                        min(self.distribution_params['weight_male']['max'] if gender == '男' else self.distribution_params['weight_female']['max'], weight))
            
            age = np.random.normal(self.distribution_params['age']['mean'], self.distribution_params['age']['std'])
            age = max(self.distribution_params['age']['min'], min(self.distribution_params['age']['max'], age))
            
            education = np.random.normal(self.distribution_params['education']['mean'], self.distribution_params['education']['std'])
            education = max(self.distribution_params['education']['min'], min(self.distribution_params['education']['max'], education))
            
            appearance = np.random.normal(self.distribution_params['appearance']['mean'], self.distribution_params['appearance']['std'])
            appearance = max(self.distribution_params['appearance']['min'], min(self.distribution_params['appearance']['max'], appearance))
            
            iq = np.random.normal(self.distribution_params['iq']['mean'], self.distribution_params['iq']['std'])
            iq = max(self.distribution_params['iq']['min'], min(self.distribution_params['iq']['max'], iq))
            
            # 对数正态分布的资产和收入
            assets_log = np.random.normal(self.distribution_params['assets']['mean'], self.distribution_params['assets']['std'])
            assets = np.exp(assets_log)
            
            income_log = np.random.normal(self.distribution_params['income']['mean'], self.distribution_params['income']['std'])
            income = np.exp(income_log)
            
            # 分类变量
            region = np.random.choice(self.categorical_dists['region'], p=self.categorical_dists['region_probs'])
            hukou = np.random.choice(self.categorical_dists['hukou'], p=self.categorical_dists['hukou_probs'])
            health = np.random.choice(self.categorical_dists['health'], p=self.categorical_dists['health_probs'])
            family_bg = np.random.choice(self.categorical_dists['family_bg'], p=self.categorical_dists['family_bg_probs'])
            
            data.append({
                'gender': gender,
                'age': round(age, 1),
                'height': round(height, 1),
                'weight': round(weight, 1),
                'region': region,
                'hukou': hukou,
                'education': round(education, 1),
                'health': health,
                'appearance': round(appearance, 1),
                'iq': round(iq, 1),
                'family_bg': family_bg,
                'assets': round(assets, 2),
                'income': round(income, 2)
            })
        
        return pd.DataFrame(data)

    def plot_probability_density(self, df, column, ax=None):
        """绘制单个变量的概率密度函数"""
        if ax is None:
            fig, ax = plt.subplots(figsize=(10, 6))
        
        if df[column].dtype in [np.int64, np.float64]:
            # 连续变量使用KDE
            sns.kdeplot(data=df, x=column, ax=ax, fill=True, alpha=0.6)
            ax.set_title(f'{column}的概率密度分布', fontsize=14, fontweight='bold')
            ax.set_xlabel(column)
            ax.set_ylabel('概率密度')
            
            # 添加统计信息
            mean_val = df[column].mean()
            std_val = df[column].std()
            ax.axvline(mean_val, color='red', linestyle='--', label=f'均值: {mean_val:.2f}')
            ax.axvline(mean_val + std_val, color='orange', linestyle='--', alpha=0.7, label=f'±1标准差')
            ax.axvline(mean_val - std_val, color='orange', linestyle='--', alpha=0.7)
            ax.legend()
        else:
            # 分类变量使用条形图
            value_counts = df[column].value_counts(normalize=True)
            value_counts.plot(kind='bar', ax=ax, color='skyblue', alpha=0.7)
            ax.set_title(f'{column}的分布', fontsize=14, fontweight='bold')
            ax.set_xlabel(column)
            ax.set_ylabel('比例')
            ax.tick_params(axis='x', rotation=45)
        
        return ax

    def plot_all_distributions(self, df):
        """绘制所有13个指标的分布图"""
        numeric_columns = ['age', 'height', 'weight', 'education', 'appearance', 'iq', 'assets', 'income']
        categorical_columns = ['gender', 'region', 'hukou', 'health', 'family_bg']
        
        # 创建子图布局
        fig, axes = plt.subplots(4, 4, figsize=(20, 16))
        axes = axes.flatten()
        
        # 绘制数值变量
        for i, col in enumerate(numeric_columns):
            if i < len(axes):
                self.plot_probability_density(df, col, axes[i])
        
        # 绘制分类变量
        for i, col in enumerate(categorical_columns):
            idx = len(numeric_columns) + i
            if idx < len(axes):
                self.plot_probability_density(df, col, axes[idx])
        
        # 隐藏多余的子图
        for i in range(len(numeric_columns) + len(categorical_columns), len(axes)):
            axes[i].set_visible(False)
        
        plt.tight_layout()
        return fig

    def plot_scatter_matrix(self, df, columns=None):
        """绘制散点图矩阵"""
        if columns is None:
            columns = ['age', 'height', 'education', 'appearance', 'iq', 'assets', 'income']
        
        # 选择数值型列
        numeric_df = df[columns].select_dtypes(include=[np.number])
        
        # 创建散点图矩阵
        fig = plt.figure(figsize=(16, 12))
        scatter_matrix = pd.plotting.scatter_matrix(numeric_df, alpha=0.6, figsize=(16, 12), diagonal='kde')
        
        # 美化图形
        for ax in scatter_matrix.ravel():
            ax.set_xlabel(ax.get_xlabel(), fontsize=8, rotation=45)
            ax.set_ylabel(ax.get_ylabel(), fontsize=8, rotation=45)
            ax.tick_params(labelsize=6)
        
        plt.suptitle('婚恋指标散点图矩阵与分布', fontsize=16, fontweight='bold')
        plt.tight_layout()
        
        return fig

    def plot_correlation_heatmap(self, df):
        """绘制相关性热力图"""
        numeric_columns = df.select_dtypes(include=[np.number]).columns
        correlation_matrix = df[numeric_columns].corr()
        
        fig, ax = plt.subplots(figsize=(12, 10))
        sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0, 
                   square=True, ax=ax, fmt='.2f', cbar_kws={'shrink': 0.8})
        ax.set_title('婚恋指标相关性热力图', fontsize=16, fontweight='bold')
        
        return fig

    def analyze_gender_differences(self, df):
        """分析性别差异"""
        numeric_columns = ['age', 'height', 'weight', 'education', 'appearance', 'iq', 'assets', 'income']
        
        fig, axes = plt.subplots(2, 4, figsize=(20, 10))
        axes = axes.flatten()
        
        for i, col in enumerate(numeric_columns):
            if i < len(axes):
                # 按性别分组绘制分布
                for gender in ['男', '女']:
                    subset = df[df['gender'] == gender]
                    sns.kdeplot(data=subset, x=col, ax=axes[i], label=gender, fill=True, alpha=0.5)
                
                axes[i].set_title(f'{col}的性别分布差异')
                axes[i].legend()
                axes[i].set_xlabel(col)
                axes[i].set_ylabel('密度')
        
        plt.suptitle('婚恋指标的性别差异分析', fontsize=16, fontweight='bold')
        plt.tight_layout()
        
        return fig

def main():
    """主函数：运行完整的可视化分析"""
    print("开始生成中国婚恋市场数据可视化...")
    
    # 创建可视化器实例
    visualizer = DatingMarketVisualizer()
    
    # 生成样本数据（基于统计局数据分布）
    print("生成模拟数据中...")
    df = visualizer.generate_sample_data(10000)
    
    print("数据基本信息：")
    print(f"数据形状: {df.shape}")
    print("\n前5行数据：")
    print(df.head())
    
    print("\n数值型变量描述性统计：")
    print(df.describe())
    
    # 1. 绘制所有分布图
    print("\n绘制概率密度函数...")
    fig1 = visualizer.plot_all_distributions(df)
    plt.show()
    
    # 2. 绘制散点图矩阵
    print("绘制散点图矩阵...")
    fig2 = visualizer.plot_scatter_matrix(df)
    plt.show()
    
    # 3. 绘制相关性热力图
    print("绘制相关性热力图...")
    fig3 = visualizer.plot_correlation_heatmap(df)
    plt.show()
    
    # 4. 分析性别差异
    print("分析性别差异...")
    fig4 = visualizer.analyze_gender_differences(df)
    plt.show()
    
    # 保存数据到CSV
    df.to_csv('chinese_dating_market_data.csv', index=False, encoding='utf-8-sig')
    print("\n数据已保存到: chinese_dating_market_data.csv")
    
    return df

if __name__ == "__main__":
    # 运行完整分析
    df = main()