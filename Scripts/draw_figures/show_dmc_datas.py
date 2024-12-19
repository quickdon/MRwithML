import numpy as np
import matplotlib.pyplot as plt

# 加载 npz 文件
data = np.load('path/DMC.npz', allow_pickle=True)

# 提取 Etot 数据
Etot = data['Etot']

# Display basic statistics of 'Etot'
etot_stats = {
    'mean': np.mean(Etot),
    'median': np.median(Etot),
    'std': np.std(Etot),
    'min': np.min(Etot),
    'max': np.max(Etot)
}

# Plotting the distribution of Etot with annotations and larger fonts
plt.figure(figsize=(9, 6))

# Plot histogram
plt.hist(Etot, bins=50, edgecolor='black', alpha=0.7, color='skyblue')
plt.axvline(etot_stats['mean'], color='red', linestyle='dashed', linewidth=2)
plt.axvline(etot_stats['median'], color='green', linestyle='dashed', linewidth=2)
plt.axvline(etot_stats['min'], color='blue', linestyle='dashed', linewidth=2)
plt.axvline(etot_stats['max'], color='orange', linestyle='dashed', linewidth=2)

# Annotations with adjusted positions and larger font size
plt.text(etot_stats['mean'], plt.ylim()[1]*0.89, 'Mean: {:.2f} Ha'.format(etot_stats['mean']), color='red', ha='right', fontsize=14)
plt.text(etot_stats['median'], plt.ylim()[1]*0.96, 'Median: {:.2f} Ha'.format(etot_stats['median']), color='green', ha='left', fontsize=14)
plt.text(etot_stats['min'], plt.ylim()[1]*0.75, 'Min: {:.2f} Ha'.format(etot_stats['min']), color='blue', ha='left', fontsize=14)
plt.text(etot_stats['max'], plt.ylim()[1]*0.65, 'Max: {:.2f} Ha'.format(etot_stats['max']), color='orange', ha='right', fontsize=14)

# Titles and labels with larger fonts
plt.xlabel('Total Energy (Ha)', fontsize=16)
plt.ylabel('Counts', fontsize=16)
plt.title('Distribution of Total Energy', fontsize=18)
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
plt.grid(True)
plt.tight_layout()

# Save the plot
plot_path_beautified = 'path/Etot_distribution.png'
plt.savefig(plot_path_beautified, dpi=300)

# Display the plot
plt.show()
