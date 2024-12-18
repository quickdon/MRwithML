import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# 读取数据
data = pd.read_csv('CSVs/datas.csv', index_col=0)

# 定义非常大的值
large_positive_value1 = 150.000
large_positive_value2 = 15.000
large_negative_value = -10.000

# 初始化空列表，用于存储拆分后的数据
rows = []

# 遍历数据框并拆分每个单元格的值
for desc, row in data.iterrows():
    for model, values in row.items():
        mse, mae, r2 = values.split()
        # 处理特殊值“==”
        if mse == '==':
            mse = large_positive_value1
        else:
            mse = float(mse)
        if mae == '==':
            mae = large_positive_value2
        else:
            mae = float(mae)
        if r2 == '==':
            r2 = large_negative_value
        else:
            r2 = float(r2)
        # 添加到行列表
        rows.append([desc, model, 'MSE', mse])
        rows.append([desc, model, 'MAE', mae])
        rows.append([desc, model, 'R2', r2])

# 创建新的数据框架
df = pd.DataFrame(rows, columns=['Descriptor', 'Model', 'Metric', 'Value'])

# 美化设置
sns.set_theme(style="whitegrid")
palette = sns.color_palette("Set2")

# 绘制模型的MSE比较箱线图
plt.figure(figsize=(8, 5.6))
sns.boxplot(x='Model', y='Value', data=df[df['Metric'] == 'MSE'], palette=palette)
plt.title('Algorithms Performance Comparison - MSE', fontsize=18)
plt.xlabel('Algorithms', fontsize=16)
plt.ylabel('MSE', fontsize=16)
plt.xticks(rotation=45, fontsize=14)
plt.yticks(fontsize=14)
plt.grid(True)
plt.tight_layout()
plt.savefig('path/Model_Performance_Comparison_MSE.png')
plt.show()


# 绘制模型的MAE比较箱线图
plt.figure(figsize=(8, 5.6))
sns.boxplot(x='Model', y='Value', data=df[df['Metric'] == 'MAE'], palette=palette)
plt.title('Algorithms Performance Comparison - MAE', fontsize=18)
plt.xlabel('Algorithms', fontsize=16)
plt.ylabel('MAE', fontsize=16)
plt.xticks(rotation=45, fontsize=14)
plt.yticks(fontsize=14)
plt.grid(True)
plt.tight_layout()
plt.savefig('path/Model_Performance_Comparison_MAE.png')
plt.show()

# 绘制模型的R2比较箱线图
plt.figure(figsize=(8, 5.6))
sns.boxplot(x='Model', y='Value', data=df[df['Metric'] == 'R2'], palette=palette)
plt.title('Algorithms Performance Comparison - R$^2$', fontsize=18)
plt.xlabel('Algorithms', fontsize=16)
plt.ylabel('R$^2$', fontsize=16)
plt.xticks(rotation=45, fontsize=14)
plt.yticks(fontsize=14)
plt.ylim(0, 1)  # 放大纵坐标
plt.axhline(y=-1, color='gray', linestyle='--')
plt.grid(True)
plt.tight_layout()
plt.savefig('path/Model_Performance_Comparison_R2.png')
plt.show()

# 绘制描述符的MSE比较箱线图
plt.figure(figsize=(8, 6))
sns.boxplot(x='Descriptor', y='Value', data=df[df['Metric'] == 'MSE'], palette=palette)
plt.title('Fingerprints Performance Comparison - MSE', fontsize=18)
plt.xlabel('Fingerprints', fontsize=16)
plt.ylabel('MSE', fontsize=16)
plt.xticks(rotation=45, fontsize=14)
plt.yticks(fontsize=14)
plt.ylim(0, 80)
plt.grid(True)
plt.tight_layout()
plt.savefig('path/Descriptor_Performance_Comparison_MSE.png')
plt.show()

# 绘制描述符的MAE比较箱线图
plt.figure(figsize=(8, 6))
sns.boxplot(x='Descriptor', y='Value', data=df[df['Metric'] == 'MAE'], palette=palette)
plt.title('Fingerprints Performance Comparison - MAE', fontsize=18)
plt.xlabel('Fingerprints', fontsize=16)
plt.ylabel('MAE', fontsize=16)
plt.xticks(rotation=45, fontsize=14)
plt.yticks(fontsize=14)
plt.ylim(0, 8)
plt.grid(True)
plt.tight_layout()
plt.savefig('path/Descriptor_Performance_Comparison_MAE.png')
plt.show()

# 绘制描述符的R2比较箱线图
plt.figure(figsize=(8, 6))
sns.boxplot(x='Descriptor', y='Value', data=df[df['Metric'] == 'R2'], palette=palette)
plt.title('Fingerprints Performance Comparison - R$^2$', fontsize=18)
plt.xlabel('Fingerprints', fontsize=16)
plt.ylabel('R$^2$', fontsize=16)
plt.xticks(rotation=45, fontsize=14)
plt.yticks(fontsize=14)
plt.ylim(0, 1)  # 放大纵坐标
plt.axhline(y=-1, color='gray', linestyle='--')
plt.grid(True)
plt.tight_layout()
plt.savefig('path/Descriptor_Performance_Comparison_R2.png')
plt.show()
