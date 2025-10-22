import random
import numpy as np
from scipy.stats import norm
import pandas as pd

class ChineseDatingMarket:
    def __init__(self):
        # 基于中国社会数据的参数设置
        self.population_stats = {
            # 性别比例 (男:女 ≈ 1.04:1)
            'gender_ratio': 0.509,  # 男性比例
            
            # 年龄分布 (适婚人群 20-40岁)
            'age_mean': 30,
            'age_std': 8,
            
            # 身高分布 (cm)
            'male_height_mean': 172,
            'male_height_std': 6,
            'female_height_mean': 160,
            'female_height_std': 5,
            
            # 体重分布 (kg)
            'male_weight_mean': 70,
            'male_weight_std': 10,
            'female_weight_mean': 55,
            'female_weight_std': 8,
            
            # 学历分布 (0-10分, 10为最高)
            'education_mean': 6.5,
            'education_std': 2.5,
            
            # 外貌评分 (0-100分)
            'appearance_mean': 60,
            'appearance_std': 15,
            
            # 智商分布
            'iq_mean': 100,
            'iq_std': 15,
            
            # 资产状况 (万元, 对数正态分布)
            'assets_log_mean': 2.5,  # 约12万元
            'assets_log_std': 1.2,
        }
        
        # 13个核心指标
        self.attributes = [
            'gender', 'age', 'height', 'weight', 'region', 'hukou', 
            'education', 'health', 'appearance', 'iq', 'family_background', 
            'assets', 'income'
        ]
    
    def generate_individual(self):
        """生成一个符合中国社会分布的个体"""
        gender = 1 if random.random() < self.population_stats['gender_ratio'] else 0  # 1=男, 0=女
        
        # 基于性别调整参数
        height_mean = self.population_stats['male_height_mean'] if gender == 1 else self.population_stats['female_height_mean']
        height_std = self.population_stats['male_height_std'] if gender == 1 else self.population_stats['female_height_std']
        weight_mean = self.population_stats['male_weight_mean'] if gender == 1 else self.population_stats['female_weight_mean']
        weight_std = self.population_stats['male_weight_std'] if gender == 1 else self.population_stats['female_weight_std']
        
        individual = {
            'gender': gender,
            'age': max(18, min(60, random.gauss(self.population_stats['age_mean'], 
                                               self.population_stats['age_std']))),
            'height': max(140, min(210, random.gauss(height_mean, height_std))),
            'weight': max(40, min(150, random.gauss(weight_mean, weight_std))),
            'region': random.randint(0, 3),  # 0=一线, 1=二线, 2=三线, 3=农村
            'hukou': random.randint(0, 3),  # 户籍等级
            'education': max(0, min(10, random.gauss(self.population_stats['education_mean'], 
                                                   self.population_stats['education_std']))),
            'health': random.uniform(0.7, 1.0),  # 健康状态
            'appearance': max(0, min(100, random.gauss(self.population_stats['appearance_mean'], 
                                                     self.population_stats['appearance_std']))),
            'iq': max(70, min(140, random.gauss(self.population_stats['iq_mean'], 
                                              self.population_stats['iq_std']))),
            'family_background': random.uniform(0, 10),  # 家庭背景评分
            'assets': np.exp(random.gauss(self.population_stats['assets_log_mean'], 
                                        self.population_stats['assets_log_std'])),  # 对数正态分布
            'income': max(3, random.gauss(15, 10))  # 月收入(千元)
        }
        
        return individual
    
    def calculate_compatibility(self, person_a, person_b):
        """计算两个人之间的匹配度 (0-1之间)"""
        if person_a['gender'] == person_b['gender']:
            return 0  # 不考虑同性关系
        
        # 各项指标的权重 (可根据实际情况调整)
        weights = {
            'age': 0.15,        # 年龄相似度
            'education': 0.12,   # 教育匹配
            'assets': 0.12,     # 资产状况
            'region': 0.10,     # 地区接近
            'appearance': 0.10,  # 外貌吸引
            'hukou': 0.08,      # 户籍匹配
            'iq': 0.08,         # 智商相近
            'family_background': 0.08,
            'income': 0.07,
            'height': 0.05,
            'weight': 0.03,
            'health': 0.02
        }
        
        total_score = 0
        max_score = sum(weights.values())
        
        # 计算年龄匹配度 (年龄差越小越好)
        age_diff = abs(person_a['age'] - person_b['age'])
        age_score = max(0, 1 - age_diff/15)  # 年龄差15岁以内有分
        
        # 教育匹配度
        edu_diff = abs(person_a['education'] - person_b['education'])
        edu_score = max(0, 1 - edu_diff/5)
        
        # 资产匹配度 (对数尺度比较)
        assets_a = max(1, person_a['assets'])
        assets_b = max(1, person_b['assets'])
        assets_ratio = min(assets_a, assets_b) / max(assets_a, assets_b)
        assets_score = assets_ratio
        
        # 地区匹配度
        region_score = 1 if person_a['region'] == person_b['region'] else 0.3
        
        # 外貌吸引 (互补性: 外貌差异不大时分数高)
        appearance_diff = abs(person_a['appearance'] - person_b['appearance'])
        appearance_score = max(0, 1 - appearance_diff/40)
        
        # 计算总分
        total_score = (weights['age'] * age_score + 
                      weights['education'] * edu_score +
                      weights['assets'] * assets_score +
                      weights['region'] * region_score +
                      weights['appearance'] * appearance_score +
                      weights['hukou'] * (1 if person_a['hukou'] == person_b['hukou'] else 0.2) +
                      weights['iq'] * max(0, 1 - abs(person_a['iq'] - person_b['iq'])/30) +
                      weights['family_background'] * max(0, 1 - abs(person_a['family_background'] - person_b['family_background'])/5) +
                      weights['income'] * max(0, 1 - abs(person_a['income'] - person_b['income'])/20) +
                      weights['height'] * max(0, 1 - abs(person_a['height'] - person_b['height'])/30) +
                      weights['weight'] * max(0, 1 - abs(person_a['weight'] - person_b['weight'])/25) +
                      weights['health'] * (person_a['health'] * person_b['health']))
        
        return total_score / max_score
    
    def monte_carlo_matching(self, num_samples, compatibility_threshold=0.6):
        """蒙特卡洛模拟婚恋匹配"""
        successful_matches = 0
        total_compatibility = 0
        match_details = []
        
        for i in range(num_samples):
            # 生成随机的男性和女性
            male = self.generate_individual()
            while male['gender'] != 1:  # 确保是男性
                male = self.generate_individual()
                
            female = self.generate_individual()  
            while female['gender'] != 0:  # 确保是女性
                female = self.generate_individual()
            
            # 计算匹配度
            compatibility = self.calculate_compatibility(male, female)
            total_compatibility += compatibility
            
            # 判断是否匹配成功
            if compatibility >= compatibility_threshold:
                successful_matches += 1
                match_details.append({
                    'male_age': male['age'],
                    'female_age': female['age'], 
                    'compatibility': compatibility,
                    'male_education': male['education'],
                    'female_education': female['education'],
                    'male_assets': male['assets'],
                    'female_assets': female['assets']
                })
        
        match_probability = successful_matches / num_samples
        avg_compatibility = total_compatibility / num_samples
        
        return match_probability, avg_compatibility, match_details

def run_analysis():
    """运行完整的蒙特卡洛分析"""
    market = ChineseDatingMarket()
    
    # 不同样本量的测试
    sample_sizes = [1000, 5000, 10000, 50000]
    results = []
    
    print("中国婚恋市场蒙特卡洛模拟分析")
    print("=" * 50)
    
    for size in sample_sizes:
        probability, avg_comp, details = market.monte_carlo_matching(size)
        results.append({
            'sample_size': size,
            'match_probability': probability,
            'avg_compatibility': avg_comp,
            'successful_matches': len(details)
        })
        
        print(f"样本量: {size:>6} | 匹配概率: {probability:.3f} | "
              f"平均匹配度: {avg_comp:.3f} | 成功匹配数: {len(details)}")
    
    # 生成一些统计信息
    if details:
        df = pd.DataFrame(details)
        print("\n匹配成功对的特征分析:")
        print(f"平均年龄差: {abs(df['male_age'] - df['female_age']).mean():.2f}岁")
        print(f"教育匹配度: {abs(df['male_education'] - df['female_education']).mean():.2f}")
        print(f"资产比率: {(df['male_assets'] / df['female_assets']).mean():.2f}")
    
    return results, details

# 运行分析
if __name__ == "__main__":
    results, match_details = run_analysis()
    
    # 保存结果到文件（可选）
    df_results = pd.DataFrame(results)
    df_results.to_csv('dating_market_results.csv', index=False)
    
    print("\n分析完成！结果已保存到 dating_market_results.csv")

# 快速测试
market = ChineseDatingMarket()
prob, avg_comp, details = market.monte_carlo_matching(10000)
print(f"匹配概率: {prob:.2%}, 平均匹配度: {avg_comp:.3f}")

# 参数敏感性分析（调整匹配阈值）
thresholds = [0.5, 0.6, 0.7, 0.8]
for threshold in thresholds:
    prob, _, _ = market.monte_carlo_matching(5000, threshold)
    print(f"阈值 {threshold}: 匹配概率 {prob:.3f}")