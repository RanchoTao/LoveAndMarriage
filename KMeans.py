import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns

class ChineseDatingMarket:
    def __init__(self, n_male=10000, n_female=10000):
        self.n_male = n_male
        self.n_female = n_female
        self.attributes = [
            'age', 'height', 'weight', 'income', 'education', 'property', 
            'appearance', 'health', 'family_bg', 'hukou', 'region', 
            'emotional_intel', 'hobby_match', 'libido'
        ]
        # 属性分布参数（基于中国现实数据校准）
        self.dist_params = {
            'age': {'type': 'normal', 'mean': 35, 'std': 8},
            'height': {'type': 'normal', 'mean': 169, 'std': 6},
            'weight': {'type': 'normal', 'mean': 65, 'std': 10},
            'income': {'type': 'exponential', 'scale': 20000, 'shift': 30000},
            'education': {'type': 'normal', 'mean': 60, 'std': 20},
            'property': {'type': 'exponential', 'scale': 500000, 'shift': 200000},
            'appearance': {'type': 'normal', 'mean': 60, 'std': 15},
            'health': {'type': 'normal', 'mean': 70, 'std': 10},
            'family_bg': {'type': 'normal', 'mean': 60, 'std': 15},
            'hukou': {'type': 'binary', 'p_urban': 0.6},  # 1=城镇, 0=农村
            'region': {'type': 'categorical', 'n_categories': 4},  # 4大区域
            'emotional_intel': {'type': 'normal', 'mean': 60, 'std': 15},
            'hobby_match': {'type': 'normal', 'mean': 50, 'std': 20},
            'libido': {'type': 'normal', 'mean': 50, 'std': 15}
        }
        
    def generate_population(self):
        """生成男性和女性人口数据"""
        populations = []
        for gender, n in [('male', self.n_male), ('female', self.n_female)]:
            data = {}
            for attr, params in self.dist_params.items():
                if params['type'] == 'normal':
                    data[attr] = np.random.normal(params['mean'], params['std'], n)
                elif params['type'] == 'exponential':
                    data[attr] = np.random.exponential(params['scale'], n) + params.get('shift', 0)
                elif params['type'] == 'binary':
                    data[attr] = np.random.binomial(1, params['p_urban'], n)
                elif params['type'] == 'categorical':
                    data[attr] = np.random.randint(0, params['n_categories'], n)
            # 对连续属性裁剪到合理范围
            for attr in ['age', 'height', 'weight', 'income', 'education', 'appearance']:
                if attr in data:
                    data[attr] = np.clip(data[attr], a_min=0, a_max=100)  # 简化裁剪
            df = pd.DataFrame(data)
            df['gender'] = gender
            populations.append(df)
        return pd.concat(populations, ignore_index=True)
    
    def assign_random_weights(self, population):
        """为每个个体随机生成14维价值观权重（Dirichlet分布确保和为1）"""
        weights = np.random.dirichlet(alpha=np.ones(len(self.attributes)), size=len(population))
        weight_df = pd.DataFrame(weights, columns=[f'weight_{attr}' for attr in self.attributes])
        population = pd.concat([population, weight_df], axis=1)
        return population
    
    def calculate_compatibility(self, person_a, person_b):
        """计算两个人之间的加权匹配度（基于所有属性差异）"""
        diff_squared = 0
        total_weight = 0
        for attr in self.attributes:
            weight = person_a[f'weight_{attr}']
            # 处理分类变量（如户籍、地区）的差异计算
            if attr in ['hukou', 'region']:
                diff = 0 if person_a[attr] == person_b[attr] else 1
            else:
                # 连续变量标准化差异
                a_val = person_a[attr]
                b_val = person_b[attr]
                max_val = max(self.dist_params[attr].get('max', 100), 1)  # 防止除零
                diff = abs(a_val - b_val) / max_val
            diff_squared += weight * (diff ** 2)
            total_weight += weight
        # 加权欧氏距离，转换为相似度（0-1）
        distance = np.sqrt(diff_squared / total_weight) if total_weight > 0 else 1
        compatibility = 1 - distance
        return compatibility
    
    def simulate_matching(self, population, compatibility_threshold=0.6, max_rounds=10):
        """模拟多轮匹配过程（类似Gale-Shapley算法的改进版）"""
        males = population[population['gender'] == 'male'].copy()
        females = population[population['gender'] == 'female'].copy()
        matches = []  # 存储匹配结果
        
        for round in range(max_rounds):
            if len(males) == 0 or len(females) == 0:
                break
                
            # 男性主动求婚：每个男性选择兼容度最高的女性
            male_choices = {}
            for _, male in males.iterrows():
                best_female_id = None
                best_score = -1
                for _, female in females.iterrows():
                    score = self.calculate_compatibility(male, female)
                    if score > best_score and score >= compatibility_threshold:
                        best_score = score
                        best_female_id = female.name
                if best_female_id is not None:
                    male_choices[male.name] = (best_female_id, best_score)
            
            # 女性选择：每个女性在所有追求者中选择最佳男性
            female_suitors = {}
            for male_id, (female_id, score) in male_choices.items():
                if female_id not in female_suitors:
                    female_suitors[female_id] = []
                female_suitors[female_id].append((male_id, score))
            
            round_matches = []
            for female_id, suitors in female_suitors.items():
                if suitors:
                    # 女性选择兼容度最高的男性
                    best_male_id, best_score = max(suitors, key=lambda x: x[1])
                    round_matches.append((best_male_id, female_id, best_score, round))
            
            # 移除已匹配的个体
            matched_male_ids = [m[0] for m in round_matches]
            matched_female_ids = [m[1] for m in round_matches]
            males = males[~males.index.isin(matched_male_ids)]
            females = females[~females.index.isin(matched_female_ids)]
            matches.extend(round_matches)
        
        return pd.DataFrame(matches, columns=['male_id', 'female_id', 'compatibility', 'round'])

    def analyze_value_archetypes(self, population, matches, n_clusters=5):
        """分析价值观原型：对成功匹配者的权重进行聚类"""
        matched_male_ids = matches['male_id'].unique()
        matched_males = population.loc[matched_male_ids]
        
        # 提取权重特征
        weight_cols = [f'weight_{attr}' for attr in self.attributes]
        X = matched_males[weight_cols].values
        
        # 标准化并聚类
        X_scaled = StandardScaler().fit_transform(X)
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        clusters = kmeans.fit_predict(X_scaled)
        
        # 分析每个簇的权重特征（价值观原型）
        matched_males['cluster'] = clusters
        archetypes = matched_males.groupby('cluster')[weight_cols].mean()
        
        # 识别每个簇最突出的权重（定义原型标签）
        archetype_labels = {}
        for cluster_id in range(n_clusters):
            cluster_weights = archetypes.loc[cluster_id]
            top_attr = cluster_weights.idxmax().replace('weight_', '')
            archetype_labels[cluster_id] = f"Cluster{cluster_id}_{top_attr}_focused"
        
        return archetypes, archetype_labels, clusters

# 运行模拟
if __name__ == "__main__":
    np.random.seed(42)  # 保证可重复性
    
    # 初始化模型
    market = ChineseDatingMarket(n_male=10000, n_female=10000)
    
    # 1. 生成人口数据
    print("生成人口数据...")
    population = market.generate_population()
    
    # 2. 分配随机权重
    print("分配随机价值观权重...")
    population = market.assign_random_weights(population)
    
    # 3. 模拟匹配
    print("开始模拟匹配...")
    matches = market.simulate_matching(population, compatibility_threshold=0.6)
    
    # 4. 分析结果
    match_rate = len(matches) / min(market.n_male, market.n_female) * 100
    print(f"匹配成功率: {match_rate:.2f}%")
    print(f"平均匹配度: {matches['compatibility'].mean():.3f}")
    print(f"匹配轮次分布: {matches['round'].value_counts().sort_index().to_dict()}")
    
    # 5. 价值观原型分析
    print("分析价值观原型...")
    archetypes, labels, clusters = market.analyze_value_archetypes(population, matches, n_clusters=5)
    
    print("\n=== 价值观原型分析结果 ===")
    for cluster_id, label in labels.items():
        top_3_weights = archetypes.loc[cluster_id].nlargest(3)
        top_attrs = [attr.replace('weight_', '') for attr in top_3_weights.index]
        print(f"{label}: 最高权重属性={top_attrs}")
    
    # 6. 可视化权重分布（示例：前5个属性）
    plt.figure(figsize=(12, 6))
    weight_cols = [f'weight_{attr}' for attr in market.attributes[:5]]
    population[weight_cols].plot(kind='box', title='前5个属性的权重分布')
    plt.xticks(rotation=45)
    plt.ylabel('权重值')
    plt.tight_layout()
    plt.show()

    # 保存详细结果（可选）
    results = {
        'population': population,
        'matches': matches,
        'archetypes': archetypes,
        'archetype_labels': labels
    }
    print("模拟完成！使用 results['变量名'] 访问详细数据。")