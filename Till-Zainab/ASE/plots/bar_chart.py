import matplotlib.pyplot as plt
import numpy as np

# Data
approaches = ['Baseline', 'Failed + Perplexity', 'Failed + Thinking']

baseline = np.array([22.7, 63.9, 38.9])
failed_perplexity = np.array([45.5, 63.9, 75.0])
failed_thinking = np.array([63.9, 69.4, 83.3])

# values = [43.3, 46.6, 74.6]
# errors = [13.856, 13.048, 7.613]

values = [baseline.mean(), failed_perplexity.mean(), failed_thinking.mean()]
errors = [baseline.std(), failed_perplexity.std(), failed_thinking.std()]


# Colors matching the original script
colors = ['tab:blue', 'tab:green', 'tab:red']  # baseline, perplexity, thinking

# Create the plot
fig, ax = plt.subplots(1, 1, figsize=(10, 6))

# Create bars with error bars
x = np.arange(len(approaches))
bars = ax.bar(x, values, color=colors, alpha=0.8, edgecolor='white', linewidth=0.5, 
              yerr=errors, capsize=5, error_kw={'linewidth': 2, 'ecolor': 'black'})

# Customize the chart
ax.set_ylabel('Average Line Coverage', fontsize=16)
ax.set_xticks(x)
ax.set_xticklabels(approaches, fontsize=14)
ax.tick_params(axis='y', labelsize=14)
ax.grid(axis='y', linestyle='--', alpha=0.7)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

# Add value labels on top of bars
# for i, bar in enumerate(bars):
#     height = bar.get_height()
#     ax.text(bar.get_x() + bar.get_width()/2., height + 0.5,
#             f'{values[i]}', ha='center', va='bottom', fontsize=14, fontweight='bold')

# Adjust layout and save
fig.tight_layout()
plt.savefig('/home/till/Desktop/ASE/plots/bar_chart.png', dpi=300, bbox_inches='tight')
plt.show()