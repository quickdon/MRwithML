import pandas as pd
df = pd.read_excel('path/dft_autogluon_results.xlsx')
# Reshape the data for grouped bar plot
df_melted = pd.melt(df, id_vars=['Unnamed: 0'], var_name='Fingerprint', value_name='R2')

# Rename the column 'Unnamed: 0' to 'Property'
df_melted.rename(columns={'Unnamed: 0': 'Property'}, inplace=True)
df_melted['Property'] = df_melted['Property'].str.replace('Cp', 'C$_p$')
df_melted['Property'] = df_melted['Property'].str.replace('Eatom', 'E$_{atom}$')
df_melted['Property'] = df_melted['Property'].str.replace('Eee', 'E$_{ee}$')
df_melted['Property'] = df_melted['Property'].str.replace('Etot', 'E$_{tot}$')
df_melted['Property'] = df_melted['Property'].str.replace('Exc', 'E$_{xc}$')
# print(df_melted.head())

# Now let's plot the grouped bar chart
import matplotlib.pyplot as plt
import seaborn as sns


# Adjust the legend position to move it away from the data
# plt.figure(figsize=(9, 6.5))
fig, ax = plt.subplots(figsize=(9, 6))
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
sns.barplot(data=df_melted, x='Fingerprint', y='R2', hue='Property')

# Increase font sizes for labels and title
plt.title('R$^2$ Values for Different Properties Across Fingerprints', fontsize=16)
plt.xlabel('Fingerprints', fontsize=14)
plt.ylabel('R$^2$', fontsize=14)

# Rotate the x-axis labels and increase their font size
plt.xticks(rotation=45, fontsize=12)
plt.yticks(fontsize=12)

# Move the legend to the right side and adjust its font size
plt.legend(title='Properties', fontsize=9, title_fontsize=10, loc='center left', bbox_to_anchor=(0.88, 0.8))

plt.tight_layout()
plt.savefig('path/R2_Values_for_Different_Properties_Across_Fingerprints.png', dpi=300)
plt.show()

