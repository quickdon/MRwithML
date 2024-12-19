import pandas as pd

# Load the Excel file to check its contents
file_path = 'path/train_set_size_change_results.xlsx'
df = pd.read_excel(file_path)

# Display the first few rows to understand its structure
# print(df.head())

import matplotlib.pyplot as plt

# Extracting data
df.set_index('Unnamed: 0', inplace=True)
df.index.name = 'Property'
print(df.head())
df.index = df.index.str.replace('Eee', 'E$_{ee}$')
df.index = df.index.str.replace('Etot', 'E$_{tot}$')
df.index = df.index.str.replace('Exc', 'E$_{xc}$')

# Adjusting the figure size and font size for labels
plt.figure(figsize=(7, 5))

for property_name in df.index:
    plt.plot(df.columns, df.loc[property_name], marker='o', label=property_name)

# Adjusting font size
plt.title('MAE vs Train Set Size for Different Properties', fontsize=16)
plt.xlabel('Train Set Size', fontsize=14)
plt.ylabel('MAE', fontsize=14)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.legend(title='Properties', fontsize=12, title_fontsize=12)
plt.grid(True)
plt.tight_layout()
plt.savefig('path/MAE_vs_Train_Set_Size.png')
# Show the updated plot
plt.show()
