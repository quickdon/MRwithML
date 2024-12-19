import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

xls = pd.ExcelFile('path/test_dft_results.xlsx')
# Load all sheets into a dictionary to inspect their structure
data_dict = {sheet: pd.read_excel(xls, sheet_name=sheet) for sheet in xls.sheet_names}

# Display one of the sheets to understand the structure of the data (for example, 'AtomPair')
# print(data_dict['AtomPair'].head())

# Transpose the data to group by properties (columns) and algorithms (rows)
# For each property, create a heatmap across all fingerprints and algorithms
def prepare_data_for_property(property_name):
    # Collect data for a specific property across all fingerprints
    property_data = {}
    for fingerprint in data_dict:
        df = data_dict[fingerprint]
        property_data[fingerprint] = df[df['Unnamed: 0'] == property_name].iloc[:, 1:].values[0]

    # Create a DataFrame with fingerprints as columns and algorithms as rows
    property_df = pd.DataFrame(property_data, index=df.columns[1:])
    return property_df

properties1 = ['C$_p$ (cal/(mol*K))', 'E$_{atom}$ (Ha)', 'E$_{ee}$ (Ha)', 'E$_{tot}$ (Ha)', 'E$_{xc}$ (Ha)']
properties = data_dict['AtomPair']['Unnamed: 0'].tolist()
# Adjust the heatmap size to be slightly smaller and increase the font size for labels
def plot_final_adjusted_heatmap(df, title):
    plt.figure(figsize=(7, 5))  # Adjust the figure size to be smaller
    masked_df = df.copy()
    masked_df[masked_df < 0] = 0
    
    sns.heatmap(masked_df, annot=True, cmap="coolwarm", vmin=0, vmax=1, fmt=".2f", cbar_kws={'label': 'R$^2$'},
                annot_kws={"size": 10},  # Increase the font size of annotations
                )
    
    plt.title(title, fontsize=14)
    plt.xlabel("Fingerprints", fontsize=12)
    plt.ylabel("Algorithms", fontsize=12)
    plt.xticks(rotation=45, ha='right', fontsize=10)  # Rotate the fingerprint labels and adjust font size
    plt.yticks(rotation=45, fontsize=10)  # Rotate the algorithm labels and adjust font size
    plt.tight_layout()
i = 0
# Plot heatmaps with the final adjustments
for property_name, property_name1 in zip(properties, properties1):
    property_df = prepare_data_for_property(property_name)
    plot_final_adjusted_heatmap(property_df, f"R$^2$ Heatmap for Property: {property_name1}")
    plt.savefig(f'path/{i}.png', dpi=300)
    plt.show()
    i += 1
