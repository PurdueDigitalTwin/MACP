import matplotlib.pyplot as plt
import seaborn as sns

# global plot style
sns.set(style='whitegrid')

methods = ['F-Cooper', 'V2VNet', 'AttFuse', 'V2X-ViT', 'CoBEVT', 'MACP']
ap_70 = [31.8, 34.3, 33.6, 36.9, 36.0, 47.9]
num_of_tunable_parameters = [7.27, 14.61, 6.58, 13.45, 10.51, 1.97]  # in millions
# total_num_of_parameters = [7.27, 14.61, 6.58, 13.45, 10.51, 8.98]  # in millions
am = [0.2, 0.2, 0.2, 0.2, 0.2, 0.13]

# Calculate marker sizes based on total number of parameters
marker_sizes = [size * 5000 for size in am]

# Define colors for bubbles
colors = ['blue', 'orange', 'green', 'purple', 'brown', 'red']

plt.figure(figsize=(12, 6))

# Scatter plot with colored bubbles and model names
for i in range(len(methods)):
    plt.scatter(num_of_tunable_parameters[i], ap_70[i], s=marker_sizes[i], alpha=0.65, c=colors[i], ec="white")

    if methods[i] == 'MACP':
        plt.annotate(r'$\mathbf{MACP}$', (num_of_tunable_parameters[i], ap_70[i]), textcoords="offset points",
                     xytext=(0, 25), ha='center', fontsize=20, weight='bold')
    else:
        plt.annotate(methods[i], (num_of_tunable_parameters[i], ap_70[i]), textcoords="offset points", xytext=(0, 25),
                     ha='center', fontsize=20)

plt.xlabel('Number of Tunable Parameters (M)', fontsize=24, weight='bold')
plt.ylabel('Average Precision (AP)', fontsize=24, weight='bold')
plt.xlim(0.0, 16.0)
plt.ylim(27.0, 53.0)  # Set y-axis range
plt.xticks(fontsize=20)
plt.yticks(range(30, 51, 5), fontsize=20)
plt.grid(True)
plt.savefig('performance_comp.png', dpi=600, bbox_inches='tight')
plt.show()
