import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import numba
from concurrent.futures import ThreadPoolExecutor, as_completed

# 将核心计算函数移到类外部，作为独立函数
@numba.jit(nopython=True, parallel=True)
def calculate_compatibility_batch(male_attrs, female_attrs, male_weights, categorical_mask):
    """使用Numba加速的批量兼容度计算 - 修复版"""
    n_male = male_attrs.shape[0]
    n_female = female_attrs.shape[0]
    n_attrs = male_attrs.shape[1]
    
    compatibility_matrix = np.zeros((n_male, n_female))
    
    for i in numba.prange(n_male):
        for j in range(n_female):
            diff_squared = 0.0
            total_weight = 0.0
            
            for k in range(n_attrs):
                weight = male_weights[i, k]
                if categorical_mask[k]:
                    # 分类变量：相等为0，不等为1
                    diff = 0.0 if male_attrs[i, k] == female_attrs[j, k] else 1.0
                else:
                    # 连续变量：标准化差异
                    max_val = 100.0  # 简化处理
                    diff = abs(male_attrs[i, k] - female_attrs[j, k]) / max_val
                
                diff_squared += weight * (diff ** 2)
                total_weight += weight
            
            if total_weight > 0:
                distance = np.sqrt(diff_squared / total_weight)
                compatibility_matrix[i, j] = 1.0 - distance
            else:
                compatibility_matrix[i, j] = 0.0
    
    return compatibility_matrix

class ChineseDatingMarketOptimized:
    def __init__(self, n_male=10000, n_female=10000):
        self.n_male = n_male
        self.n_female = n_female
        self.attributes = [
            'age', 'height', 'weight', 'income', 'education', 'property', 
            'appearance', 'health', 'family_bg', 'hukou', 'region', 
            'emotional_intel', 'hobby_match', 'libido'
        ]
        
        # 属性分布参数
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
            'hukou': {'type': 'binary', 'p_urban': 0.6},
            'region': {'type': 'categorical', 'n_categories': 4},
            'emotional_intel': {'type': 'normal', 'mean': 60, 'std': 15},
            'hobby_match': {'type': 'normal', 'mean': 50, 'std': 20},
            'libido': {'type': 'normal', 'mean': 50, 'std': 15}
        }
        
    def generate_population_optimized(self):
        """优化的数据生成方法"""
        print("生成人口数据...")
        n_total = self.n_male + self.n_female
        
        # 预分配数组
        data = {}
        for attr, params in self.dist_params.items():
            if params['type'] == 'normal':
                data[attr] = np.random.normal(params['mean'], params['std'], n_total)
            elif params['type'] == 'exponential':
                data[attr] = np.random.exponential(params['scale'], n_total) + params.get('shift', 0)
            elif params['type'] == 'binary':
                data[attr] = np.random.binomial(1, params['p_urban'], n_total)
            elif params['type'] == 'categorical':
                data[attr] = np.random.randint(0, params['n_categories'], n_total)
        
        # 裁剪连续变量
        for attr in ['age', 'height', 'weight', 'income', 'education', 'appearance']:
            if attr in data:
                data[attr] = np.clip(data[attr], 0, 100)
        
        # 创建DataFrame
        df = pd.DataFrame(data)
        df['gender'] = ['male'] * self.n_male + ['female'] * self.n_female
        return df
    
    def assign_random_weights_optimized(self, population):
        """优化的权重分配"""
        print("分配随机价值观权重...")
        n_individuals = len(population)
        weights = np.random.dirichlet(np.ones(len(self.attributes)), size=n_individuals)
        
        weight_cols = [f'weight_{attr}' for attr in self.attributes]
        for i, col in enumerate(weight_cols):
            population[col] = weights[:, i]
        
        return population
    
    def simulate_matching_optimized(self, population, compatibility_threshold=0.6, max_rounds=10, chunk_size=1000):
        """修复的匹配算法"""
        print("开始优化匹配过程...")
        
        # 分离男女性数据
        males = population[population['gender'] == 'male'].copy()
        females = population[population['gender'] == 'female'].copy()
        
        # 提取属性矩阵
        male_attrs = males[[attr for attr in self.attributes]].values
        female_attrs = females[[attr for attr in self.attributes]].values
        male_weights = males[[f'weight_{attr}' for attr in self.attributes]].values
        
        # 创建分类变量掩码
        categorical_mask = np.array([attr in ['hukou', 'region'] for attr in self.attributes])
        
        matches = []
        male_indices = np.arange(len(males))
        female_indices = np.arange(len(females))
        
        pbar = tqdm(total=max_rounds, desc="匹配轮次")
        
        for round_num in range(max_rounds):
            if len(male_indices) == 0 or len(female_indices) == 0:
                break
            
            # 分批处理以避免内存溢出
            compatibility_scores = []
            
            for i in range(0, len(male_indices), chunk_size):
                chunk_end = min(i + chunk_size, len(male_indices))
                male_chunk = male_indices[i:chunk_end]
                
                # 计算兼容度矩阵 - 修复参数传递
                male_attrs_chunk = male_attrs[male_chunk]
                male_weights_chunk = male_weights[male_chunk]
                
                # 修复：使用独立的函数而不是类方法
                compat_chunk = calculate_compatibility_batch(
                    male_attrs_chunk, 
                    female_attrs[female_indices], 
                    male_weights_chunk, 
                    categorical_mask
                )
                compatibility_scores.append(compat_chunk)
            
            # 合并所有分块结果
            if compatibility_scores:
                compatibility_matrix = np.vstack(compatibility_scores)
            else:
                compatibility_matrix = np.array([])
            
            if compatibility_matrix.size == 0:
                break
            
            # 找到超过阈值的匹配
            threshold_mask = compatibility_matrix >= compatibility_threshold
            valid_male_idx, valid_female_idx = np.where(threshold_mask)
            valid_scores = compatibility_matrix[threshold_mask]
            
            if len(valid_male_idx) == 0:
                break
            
            # 创建匹配候选
            match_candidates = []
            for i in range(len(valid_male_idx)):
                male_global_idx = male_indices[valid_male_idx[i]]
                female_global_idx = female_indices[valid_female_idx[i]]
                score = valid_scores[i]
                match_candidates.append((male_global_idx, female_global_idx, score))
            
            # 按女性分组，每个女性选择最佳男性
            female_best_matches = {}
            for male_idx, female_idx, score in match_candidates:
                if female_idx not in female_best_matches or score > female_best_matches[female_idx][1]:
                    female_best_matches[female_idx] = (male_idx, score)
            
            # 形成最终匹配
            round_matches = []
            for female_idx, (male_idx, score) in female_best_matches.items():
                round_matches.append((male_idx, female_idx, score, round_num))
            
            # 移除已匹配的个体
            matched_males = set(match[0] for match in round_matches)
            matched_females = set(match[1] for match in round_matches)
            
            male_indices = np.array([idx for idx in male_indices if idx not in matched_males])
            female_indices = np.array([idx for idx in female_indices if idx not in matched_females])
            
            matches.extend(round_matches)
            pbar.update(1)
            pbar.set_description(f"轮次 {round_num+1}: 匹配 {len(round_matches)} 对")
        
        pbar.close()
        
        return pd.DataFrame(matches, columns=['male_id', 'female_id', 'compatibility', 'round'])
    
    def analyze_value_archetypes(self, population, matches, n_clusters=5):
        """分析价值观原型"""
        print("分析价值观原型...")
        
        matched_male_ids = matches['male_id'].unique()
        matched_males = population.loc[matched_male_ids]
        
        weight_cols = [f'weight_{attr}' for attr in self.attributes]
        X = matched_males[weight_cols].values
        
        # 标准化并聚类
        X_scaled = StandardScaler().fit_transform(X)
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        clusters = kmeans.fit_predict(X_scaled)
        
        # 分析每个簇
        matched_males = matched_males.copy()
        matched_males['cluster'] = clusters
        archetypes = matched_males.groupby('cluster')[weight_cols].mean()
        
        # 识别原型标签
        archetype_labels = {}
        for cluster_id in range(n_clusters):
            cluster_weights = archetypes.loc[cluster_id]
            top_attr = cluster_weights.idxmax().replace('weight_', '')
            archetype_labels[cluster_id] = f"Cluster{cluster_id}_{top_attr}_focused"
        
        return archetypes, archetype_labels, clusters

# 运行修复后的模拟
if __name__ == "__main__":
    import time
    
    print("=== 修复版中国婚恋市场模拟 ===")
    start_time = time.time()
    
    # 初始化模型 - 先用较小规模测试
    market = ChineseDatingMarketOptimized(n_male=2000, n_female=2000)
    
    # 1. 生成人口数据
    population = market.generate_population_optimized()
    
    # 2. 分配随机权重
    population = market.assign_random_weights_optimized(population)
    
    # 3. 模拟匹配（使用修复算法）
    matches = market.simulate_matching_optimized(
        population, 
        compatibility_threshold=0.6,
        max_rounds=10,
        chunk_size=500
    )
    
    # 4. 分析结果
    match_rate = len(matches) / min(market.n_male, market.n_female) * 100
    elapsed_time = time.time() - start_time
    
    print(f"\n=== 模拟结果 ===")
    print(f"总耗时: {elapsed_time:.2f} 秒")
    print(f"匹配成功率: {match_rate:.2f}%")
    if len(matches) > 0:
        print(f"平均匹配度: {matches['compatibility'].mean():.3f}")
        print(f"匹配轮次分布: {matches['round'].value_counts().sort_index().to_dict()}")
    else:
        print("本次模拟没有成功匹配")
    
    # 5. 价值观原型分析（只有足够匹配时才进行）
    if len(matches) > 100:
        archetypes, labels, clusters = market.analyze_value_archetypes(population, matches, n_clusters=5)
        
        print("\n=== 价值观原型分析结果 ===")
        for cluster_id, label in labels.items():
            top_3_weights = archetypes.loc[cluster_id].nlargest(3)
            top_attrs = [attr.replace('weight_', '') for attr in top_3_weights.index]
            print(f"{label}: 最高权重属性={top_attrs}")
    
    print(f"\n模拟完成！耗时 {elapsed_time:.2f} 秒")