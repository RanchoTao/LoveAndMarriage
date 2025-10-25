import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np # 用于生成示例数据

# 设置中文字体，避免乱码
plt.rcParams['font.sans-serif'] = ['SimHei']  # 如果用SimHei黑体不生效，可以尝试'Kaiti'（楷体）
plt.rcParams['axes.unicode_minus'] = False   # 解决负号显示为方块的问题

# 定义行和列的标签（根据您的描述调整）
# 这些将作为热力图的“甲”和“乙”
row_labels = ["双亲不在", "双亲关系和睦", "双亲关系不和睦"]  # 甲的状态
col_labels = ["双亲健在和睦", "双亲健在不和睦", "双亲健在分离", "单亲健在"]  # 乙的状态

# 创建一个空的DataFrame，行索引为甲的状态，列名为乙的状态
df = pd.DataFrame(index=row_labels, columns=col_labels)

# 填入模拟数据（这里用随机数代替，请替换为您的实际数据）
# 例如，这个值可以表示在“甲”为“双亲不在”且“乙”为“双亲健在和睦”这种组合下的计数或分数
np.random.seed(42) # 设置随机种子以保证示例数据可重现
simulated_data = np.random.randint(low=0, high=100, size=(len(row_labels), len(col_labels)))
# 将模拟数据填入DataFrame
for i, row_label in enumerate(row_labels):
    for j, col_label in enumerate(col_labels):
        df.loc[row_label, col_label] = simulated_data[i, j]

# 确保数据类型为数值型（热力图需要）
df = df.astype(float)

print("创建的数据框如下：")
print(df)

# 创建图形
plt.figure(figsize=(10, 6))

# 使用seaborn绘制热力图
# `data=df` 指定数据源
# `annot=True` 在每个格子里显示数值
# `fmt=".0f"` 设置数值显示格式为不带小数点的整数
# `cmap='YlGnBu'` 设置颜色映射，这是一个从黄色到蓝绿色的渐变色，很常用
heatmap = sns.heatmap(data=df,
                      annot=True,
                      fmt=".0f",
                      cmap='YlGnBu',
                      linewidths=0.5,   # 设置格子间的线宽
                      linecolor='white' # 设置格子间线的颜色
                     )

# 添加标题和轴标签
plt.title("家庭背景关系热力图", fontsize=16, pad=20)
plt.xlabel("乙：家庭背景状态", fontsize=12, labelpad=10)
plt.ylabel("甲：家庭背景状态", fontsize=12, labelpad=10)

# 自动调整布局并显示图形
plt.tight_layout()
plt.show()

plt.figure(figsize=(11, 7))
# 使用mask参数隐藏值为0的格子（如果需要）
# mask = df == 0
heatmap = sns.heatmap(data=df,
                      annot=True,
                      fmt=".0f",
                      cmap='coolwarm',  # 换一个颜色方案
                      center=50,        # 将颜色中心设置为50，便于区分高低值
                      linewidths=0.75,
                      linecolor='grey',
                      cbar_kws={'label': '关系强度指标', 'shrink': 0.8},
                      # mask=mask  # 如果需要隐藏特定值（如0），取消这行的注释
                     )

plt.title("家庭背景关系分析热力图", fontsize=18, pad=20)
plt.xlabel("乙的状态", fontsize=14)
plt.ylabel("甲的状态", fontsize=14)

# 可以进一步调整X轴和Y轴标签的旋转角度，如果标签较长的话
# heatmap.set_xticklabels(heatmap.get_xticklabels(), rotation=45, ha='right')
# heatmap.set_yticklabels(heatmap.get_yticklabels(), rotation=0)

plt.tight_layout()
plt.savefig('family_background_heatmap.png', dpi=300, bbox_inches='tight') # 保存高分辨率图片
plt.show()

# 假设您有实际数据，创建一个二维列表或数组
real_data = [
    [10, 25, 5, 30],   # 当甲为"双亲不在"时，对应乙各种情况的值
    [40, 15, 20, 60],  # 当甲为"双亲关系和睦"时
    [8, 12, 35, 18]    # 当甲为"双亲关系不和睦"时
]

df_real = pd.DataFrame(real_data, index=row_labels, columns=col_labels)
# 然后用 df_real 替代上面的 df 进行绘图