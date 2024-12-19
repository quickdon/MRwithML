import numpy as np
import matplotlib.pyplot as plt

# 加载 npz 文件
data = np.load('path/DFT_uniques.npz', allow_pickle=True)

# 提取 Etot 数据
# Etot = data['Etot']
# atoms_list = data['atoms']
# total_atoms = [sum(atoms) for atoms in atoms_list]
average_Etot = data['Cp']

# Display basic statistics of 'Etot'
etot_stats = {
    'mean': np.mean(average_Etot),
    'median': np.median(average_Etot),
    'std': np.std(average_Etot),
    'min': np.min(average_Etot),
    'max': np.max(average_Etot)
}

# Plotting the distribution of Etot with annotations and larger fonts
plt.figure(figsize=(8, 6))

# Plot histogram
plt.hist(average_Etot, bins=50, edgecolor='black', alpha=0.7, color='skyblue')
plt.axvline(etot_stats['mean'], color='red', linestyle='dashed', linewidth=2)
plt.axvline(etot_stats['median'], color='green', linestyle='dashed', linewidth=2)
plt.axvline(etot_stats['min'], color='blue', linestyle='dashed', linewidth=2)
plt.axvline(etot_stats['max'], color='orange', linestyle='dashed', linewidth=2)

# Annotations with adjusted positions and larger font size
plt.text(etot_stats['mean'], plt.ylim()[1]*0.85, 'Mean: {:.2f} cal/(mol*K)'.format(etot_stats['mean']), color='red', ha='left', fontsize=14)
plt.text(etot_stats['median'], plt.ylim()[1]*0.95, 'Median: {:.2f} cal/(mol*K)'.format(etot_stats['median']), color='green', ha='right', fontsize=14)
plt.text(etot_stats['min'], plt.ylim()[1]*0.75, 'Min: {:.2f} cal/(mol*K)'.format(etot_stats['min']), color='blue', ha='left', fontsize=14)
plt.text(etot_stats['max'], plt.ylim()[1]*0.65, 'Max: {:.2f} cal/(mol*K)'.format(etot_stats['max']), color='orange', ha='right', fontsize=14)

# Titles and labels with larger fonts
plt.xlabel('Heat Capacity (C$_p$) (cal/(mol*K))', fontsize=16)
plt.ylabel('Counts', fontsize=16)
plt.title('Distribution of Heat Capacity (C$_p$)', fontsize=18)
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
plt.grid(True)
plt.tight_layout()

# Save the plot
plot_path_beautified = 'path/DFT_Cp_distribution_uniques1.png'
plt.savefig(plot_path_beautified, dpi=300)

# Display the plot
plt.show()
