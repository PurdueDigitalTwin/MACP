import seaborn as sns
import matplotlib.pyplot as plt

# Set the Seaborn style
sns.set(style='whitegrid')

compression_factor = [1, 2, 3, 4]
ap_70 = [49.6, 49.5, 49.4, 47.9]
short_ap_70 = [63.1, 63.6, 63.5, 62.1]
mid_ap_70 = [40.0, 36.9, 37.8, 38.5]
long_ap_70 = [26.5, 27.6, 25.6, 23.1]

# Create a figure and axis
plt.figure(figsize=(10, 6))
ax = plt.gca()

# Line plot for "Overall" data
line = ax.plot(
    compression_factor, ap_70,
    marker='o',
    label='Overall',
    color='black',
    linewidth=3)

# Adding annotations to the line plot
for x, y in zip(compression_factor, ap_70):
    ax.annotate(f'{y:.1f}', (x, y), textcoords="offset points", xytext=(0, 10), ha='center', fontsize=18, color='black')

# Bar plots for object ranges
bar_width = 0.2
ax.bar(
    [x - bar_width for x in compression_factor], short_ap_70,
    width=bar_width,
    label='0-30m',
    color='#b6e6bd')
ax.bar(
    compression_factor, mid_ap_70,
    width=bar_width,
    label='30-50m',
    color='#f1f0cf')
ax.bar(
    [x + bar_width for x in compression_factor], long_ap_70,
    width=bar_width,
    label='50-100m',
    color='#f0c9c9')

# Set x-axis labels and ticks
# ax.set_xscale('log')
ax.set_xticks(compression_factor, [str(4 ** x) for x in compression_factor])
ax.set_xlabel('Compression Factor', fontweight='bold', fontsize=24)

# Set y-axis label and range
ax.set_ylabel('AP@IOU=70', fontweight='bold', fontsize=24)
ax.set_ylim(0, 70)

# Set y-axis ticks
ax.set_yticks(range(0, 71, 10))
ax.tick_params(axis='both', labelsize=20)

# Add legend with adjusted fontsize and position
legend = ax.legend(
    title='Object Range', fontsize=18, title_fontsize=18, loc='lower left')

# Set title fontsize to 12
plt.setp(legend.get_title(), fontsize=20)

# Add grid
ax.grid(False, axis='x')

# Save the figure and display the plot
plt.tight_layout()
plt.savefig('compression_factor.png', dpi=600, bbox_inches='tight')
plt.show()
