import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

# --- Create the DataFrame (same as before) ---
iter_levels = ['1 iter', '2 iter', '4 iter', '6 iter']
percentage_levels = ['30%', '50%', '70%']
attempt_levels = ['Baseline', 'Failed + Thinking', 'Failed + Perplexity']
row_tuples = [(att, it, per) for att in attempt_levels for it in iter_levels for per in percentage_levels]
row_index = pd.MultiIndex.from_tuples(row_tuples, names=['Approach', 'Iterations', 'Percentage'])

metric_levels = ['Coverage', 'LLM Calls', 'Number of Tests Generated']
case_levels = ['5 cases', '10 cases', '15 cases']
column_tuples = [(met, case) for met in metric_levels for case in case_levels]
column_index = pd.MultiIndex.from_tuples(column_tuples, names=['Metric', 'Cases'])

import numpy as np

data = np.array([
    # Baseline
    # 1 iter
    [94.7, 72.4, 66.7, 4, 4, 4, 4, 4, 4],
    [94.7, 93.1, 100.0, 6, 10, 16, 6, 10, 16],
    [94.7, 100.0, 69.2, 6, 11, 5, 6, 11, 5],
    # 2 iter
    [84.2, 72.4, 66.7, 3, 4, 4, 3, 4, 4],
    [94.7, 93.1, 89.7, 5, 10, 13, 5, 10, 13],
    [94.7, 100.0, 69.2, 6, 11, 5, 6, 11, 5],
    # 4 iter
    [84.2, 72.4, 66.7, 4, 4, 4, 4, 4, 4],
    [94.7, 93.1, 94.9, 6, 10, 15, 6, 10, 15],
    [94.7, 100.0, 69.2, 6, 11, 5, 6, 11, 5],
    # 6 iter
    [94.7, 72.4, 66.7, 5, 4, 4, 5, 4, 4],
    [84.2, 93.1, 89.7, 3, 10, 12, 3, 10, 12],
    [94.7, 100.0, 69.2, 6, 11, 5, 6, 11, 5],
    
    # Failed, Thinking
    # 1 iter
    [89.5, 72.4, 66.7, 6, 4, 5, 6, 4, 5],
    [94.7, 93.1, 94.9, 7, 11, 16, 7, 11, 16],
    [94.7, 93.1, 69.2, 7, 11, 6, 7, 11, 6],
    # 2 iter
    [100.0, 72.4, 66.7, 7, 6, 5, 7, 6, 5],
    [100.0, 93.1, 94.9, 6, 12, 17, 6, 12, 17],
    [100.0, 100.0, 69.2, 7, 11, 7, 7, 11, 7],
    # 4 iter
    [100.0, 100.0, 94.9, 7, 15, 24, 5, 10, 15],
    [100.0, 93.1, 100.0, 8, 14, 21, 6, 10, 16],
    [100.0, 100.0, 69.2, 8, 11, 5, 6, 11, 9],
    # 6 iter
    [100.0, 100.0, 97.4, 6, 15, 25, 6, 12, 16],
    [100.0, 93.1, 100.0, 6, 16, 21, 6, 10, 17],
    [100.0, 100.0, 97.4, 7, 11, 34, 7, 12, 16],

    
    # Data Format: Coverage (5cases, 10cases, 15cases), LLM Calls (5cases, 10cases, 15cases), Num Tests (5cases, 10cases, 15cases)
    # Failed, Perplexity
    # 1 iter
    [84.2, 72.4, 66.7, 71, 87, 101, 3, 4, 4],
    [94.7, 93.1, 89.7, 76, 99, 113, 5, 10, 13],
    [94.7, 100.0, 53.8, 79, 11, 98, 6, 11, 2],
    # 2 iter
    [89.5, 72.4, 66.7, 134, 157, 182, 5, 4, 4],
    [94.7, 93.1, 94.9, 136, 166, 296, 5, 8, 15],
    [94.7, 100.0, 69.2, 140, 11, 182, 6, 12, 5],
    # 4 iter
    [89.5, 72.4, 66.7, 265, 317, 373, 4, 4, 4],
    [100.0, 93.1, 100.0, 147, 240, 312, 6, 9, 17],
    [94.7, 100.0, 69.2, 272, 11, 367, 6, 12, 5],
    # 6 iter
    [89.5, 72.4, 66.7, 379, 543, 533, 5, 4, 4],
    [100.0, 93.1, 100.0, 138, 628, 410, 6, 10, 17],
    [94.7, 100.0, 69.2, 397, 11, 545, 6, 12, 5],
])
df = pd.DataFrame(data, index=row_index, columns=column_index)

# --- Common plotting setup ---
labels = ['5 cases', '10 cases', '15 cases']
x = np.arange(len(labels))
width = 0.25  # Reduced width to accommodate 3 groups

# Define colors for the three groups and alpha levels for stacking
color_baseline = 'tab:blue'
color_thinking = 'tab:red'
color_perplexity = 'tab:green'
alphas = [0.4, 0.6, 0.8, 1.0] # Increasing opacity for more iterations

# --- CHART 1: LLM Calls ---
fig1, ax1 = plt.subplots(1, 1, figsize=(14, 8))
df_llm_calls = df['LLM Calls'].copy()
df_agg_llm = df_llm_calls.groupby(level=['Approach', 'Iterations']).mean()
iters = df_agg_llm.index.get_level_values('Iterations').unique()

# Initialize the bottom of the bars for stacking and previous sums for delta calc
bottom_baseline_llm = np.zeros(len(labels))
bottom_thinking_llm = np.zeros(len(labels))
bottom_perplexity_llm = np.zeros(len(labels))
prev_baseline = np.zeros(len(labels))
prev_thinking = np.zeros(len(labels))
prev_perplexity = np.zeros(len(labels))

# Loop through each iteration level to create the stacked bars
for i, iter_level in enumerate(iters):
    values_baseline = df_agg_llm.loc[('Baseline', iter_level)].values
    values_thinking = df_agg_llm.loc[('Failed + Thinking', iter_level)].values
    values_perplexity = df_agg_llm.loc[('Failed + Perplexity', iter_level)].values

    # Compute deltas relative to previous cumulative sums, clamp at zero
    deltas_baseline = np.maximum(0, values_baseline - prev_baseline)
    deltas_thinking = np.maximum(0, values_thinking - prev_thinking)
    deltas_perplexity = np.maximum(0, values_perplexity - prev_perplexity)

    # Plot deltas for stacking - reordered: Baseline, Perplexity, Thinking
    ax1.bar(x - width, deltas_baseline, width, bottom=bottom_baseline_llm,
            color=color_baseline, alpha=alphas[i], edgecolor='white', linewidth=0.5)
    ax1.bar(x, deltas_perplexity, width, bottom=bottom_perplexity_llm,
            color=color_perplexity, alpha=alphas[i], edgecolor='white', linewidth=0.5)
    ax1.bar(x + width, deltas_thinking, width, bottom=bottom_thinking_llm,
            color=color_thinking, alpha=alphas[i], edgecolor='white', linewidth=0.5)

    # Update bottoms and set previous sums to current iteration's totals
    bottom_baseline_llm += deltas_baseline
    bottom_thinking_llm += deltas_thinking
    bottom_perplexity_llm += deltas_perplexity
    prev_baseline = values_baseline
    prev_thinking = values_thinking
    prev_perplexity = values_perplexity

# Customize LLM Calls chart
ax1.set_ylabel('Average LLM Calls\n(across 30%, 50%, 70% levels)', fontsize=16)
# ax1.set_title('Average LLM Calls: Without vs. With Failed Attempts', fontsize=16, pad=20)
ax1.set_xticks(x)
ax1.set_xticklabels(labels, fontsize=14)
ax1.set_yscale('log')  # Set logarithmic y-axis
ax1.tick_params(axis='y', labelsize=14)  # Set y-axis tick label size
ax1.grid(axis='y', linestyle='--', alpha=0.7)
ax1.spines['top'].set_visible(False)
ax1.spines['right'].set_visible(False)

# Create legends for both charts
legend1_elements = [Patch(facecolor=color_baseline, label='Baseline'),
                    Patch(facecolor=color_perplexity, label='Failed + Perplexity'),
                    Patch(facecolor=color_thinking, label='Failed + Thinking')]
legend2_elements = [Patch(facecolor='grey', alpha=alphas[i], label=iter_level) for i, iter_level in enumerate(iters)]

# Add legends to the first chart
legend1_ax1 = ax1.legend(handles=legend1_elements, loc='upper left', title='Approach', fontsize=12, title_fontsize=13)
legend2_ax1 = ax1.legend(handles=legend2_elements, loc='upper right', title='Iterations (by Opacity)', fontsize=12, title_fontsize=13)
ax1.add_artist(legend1_ax1)

# Save the first plot
fig1.tight_layout()
plt.figure(fig1)
plt.savefig('plots/llm_calls_comparison.png', dpi=300, bbox_inches='tight')
plt.close(fig1)

# --- CHART 2: Coverage (Three Line Charts) ---
fig2, (ax2_baseline, ax2_perplexity, ax2_thinking) = plt.subplots(1, 3, figsize=(18, 6))

# Prepare data for line charts
df_coverage = df['Coverage'].copy()
df_agg_coverage = df_coverage.groupby(level=['Approach', 'Iterations']).mean()

# Prepare data for line chart - reshape to have iterations on x-axis
iter_numbers = [1, 2, 4, 6]  # Extract numbers from iter_levels for x-axis
case_counts = ['5 cases', '10 cases', '15 cases']
# Use difficulty-based colors: red=15, green=5, orange=10
difficulty_colors = ['green', 'orange', 'red']  # 5 cases, 10 cases, 15 cases

# Plot lines for each approach in separate subplots
approaches = ['Baseline', 'Failed + Perplexity', 'Failed + Thinking']
axes = [ax2_baseline, ax2_perplexity, ax2_thinking]
approach_titles = ['Baseline', 'Failed + Perplexity', 'Failed + Thinking']

for ax_idx, (approach, ax, title) in enumerate(zip(approaches, axes, approach_titles)):
    for i, case_count in enumerate(case_counts):
        # Get data for this case count across iterations
        values = []
        for iter_level in iters:
            values.append(df_agg_coverage.loc[(approach, iter_level), case_count])
        
        # Plot line with difficulty-based color
        ax.plot(iter_numbers, values, color=difficulty_colors[i], 
                marker='o', linewidth=3, markersize=8, label=case_count)
    
    # Customize each subplot
    ax.set_ylabel('Coverage %\n(across 30%, 50%, 70% levels)' if ax_idx == 0 else '', fontsize=14)
    ax.set_xlabel('Number of Iterations', fontsize=14)
    ax.set_title(title, fontsize=16, pad=15)
    ax.set_xticks(iter_numbers)
    ax.set_xticklabels(iter_numbers, fontsize=12)
    ax.tick_params(axis='y', labelsize=12)
    ax.grid(True, linestyle='--', alpha=0.7)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.set_ylim(65, 104)  # Set y-axis limit for percentage data
    
    # Add legend only to the rightmost plot
    if ax_idx == 2:
        ax.legend(loc='center left', bbox_to_anchor=(1, 0.5), fontsize=12)

# Save the coverage plots
fig2.tight_layout()
plt.figure(fig2)
plt.savefig('plots/coverage_trends.png', dpi=300, bbox_inches='tight')
plt.close(fig2)

print("Plots saved as:")
print("- plots/llm_calls_comparison.png")
print("- plots/coverage_trends.png")