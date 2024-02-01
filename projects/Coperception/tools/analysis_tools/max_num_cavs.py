import seaborn as sns
import matplotlib.pyplot as plt

# Set the Seaborn style
sns.set(style='whitegrid')

max_num_CAVs = [1, 2, 3, 4, 5]
ap_70 = [0.6585, 0.8104, 0.8721, 0.8862, 0.8955]
short_ap_70 = [0.9106, 0.9649, 0.9751, 0.9754, 0.9754]
mid_ap_70 = [0.6606, 0.8304, 0.8943, 0.9002, 0.9015]
long_ap_70 = [0.3489, 0.6109, 0.7395, 0.7760, 0.7964]

# Create a figure and axis
plt.figure(figsize=(12, 6))
ax = plt.gca()

# Line plot for "Overall" data
ax.plot(
    max_num_CAVs, [100 * x for x in ap_70],
    marker='o',
    label='Overall',
    color='black',
    linewidth=3)

# Adding annotations to the line plot
for x, y in zip(max_num_CAVs, [100 * x for x in ap_70]):
    ax.annotate(f'{y:.1f}', (x, y),
                textcoords="offset points", xytext=(0, 15), ha='center', fontsize=18, color='black')

# Bar plots for object ranges
bar_width = 0.2
ax.bar(
    [x - bar_width for x in max_num_CAVs], [100 * x for x in short_ap_70],
    width=bar_width,
    label='0-30m',
    color='#b6e6bd')
ax.bar(
    max_num_CAVs, [100 * x for x in mid_ap_70],
    width=bar_width,
    label='30-50m',
    color='#f1f0cf')
ax.bar(
    [x + bar_width for x in max_num_CAVs], [100 * x for x in long_ap_70],
    width=bar_width,
    label='50-100m',
    color='#f0c9c9')

# Set x-axis labels and ticks
ax.set_xticks(max_num_CAVs)
ax.set_xlabel('Max Number of CAVs', fontweight='bold', fontsize=24)

# Set y-axis label and range
ax.set_ylabel('AP@IOU=70', fontweight='bold', fontsize=24)
ax.set_ylim(30, 100)

# Set y-axis ticks
ax.set_yticks(range(40, 101, 10))
ax.tick_params(axis='both', labelsize=20)

# Add legend with adjusted fontsize and position
legend = ax.legend(
    title='Object Range', fontsize=18, title_fontsize=18, loc='lower right')

# Set title fontsize to 18
plt.setp(legend.get_title(), fontsize=20)

# Add grid
ax.grid(False, axis='x')

# Save the figure and display the plot
plt.tight_layout()
plt.savefig('max_num_cavs.png', dpi=600, bbox_inches='tight')
plt.show()
