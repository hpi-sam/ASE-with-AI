import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

# --- Create the DataFrame (updated for three methods) ---
iter_levels = ['1 iter', '2 iter', '4 iter', '6 iter']
percentage_levels = ['30%', '50%', '70%']
attempt_levels = ['Without Failed Attempts', 'With Failed Attempts + Thinking', 'With Failed Attempts + Perplexity']
row_tuples = [(att, it, per) for att in attempt_levels for it in iter_levels for per in percentage_levels]
row_index = pd.MultiIndex.from_tuples(row_tuples, names=['Method', 'Iterations', 'Percentage'])

metric_levels = ['Coverage', 'LLM Calls', 'Number of Tests Generated']
case_levels = ['5 cases', '10 cases', '15 cases']
column_tuples = [(met, case) for met in metric_levels for case in case_levels]
column_index = pd.MultiIndex.from_tuples(column_tuples, names=['Metric', 'Cases'])

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

# --- Helper function to find Pareto front ---
def find_pareto_front(llm_calls, coverage):
    """Build Pareto front by:
    1) grouping all points by number of calls (x) and taking the max coverage per x,
    2) filtering these aggregated points to only non-dominated points, and
    3) returning them sorted by x to allow connecting neighbors.
    Note: x is negative LLM calls, so larger x means fewer actual calls (better).
    """
    points = list(zip(llm_calls, coverage))
    if not points:
        return []

    # Step 1: aggregate by x with max y
    calls_to_max_coverage = {}
    for x, y in points:
        if x in calls_to_max_coverage:
            calls_to_max_coverage[x] = max(calls_to_max_coverage[x], y)
        else:
            calls_to_max_coverage[x] = y

    aggregated_points = [(x, y) for x, y in calls_to_max_coverage.items()]

    # Step 2: filter to Pareto-optimal (non-dominated) among the aggregated points
    pareto_points = []
    for i, (x1, y1) in enumerate(aggregated_points):
        dominated = False
        for j, (x2, y2) in enumerate(aggregated_points):
            if i == j:
                continue
            # A point (x1, y1) is dominated if there exists (x2, y2) with
            # x2 >= x1 (fewer/equal calls) and y2 >= y1 (higher/equal coverage),
            # with at least one strict improvement.
            if (x2 >= x1 and y2 >= y1) and (x2 > x1 or y2 > y1):
                dominated = True
                break
        if not dominated:
            pareto_points.append((x1, y1))

    # Step 3: sort by x for sequential connection
    pareto_points.sort(key=lambda p: p[0])
    return pareto_points

# --- Create Pareto Scatter Plots (one per case group) ---
# Extract coverage and LLM calls data
df_coverage = df['Coverage'].copy()
df_llm_calls = df['LLM Calls'].copy()

# Define colors for the three case groups
color_5_cases = 'tab:green'
color_10_cases = 'tab:orange'
color_15_cases = 'tab:red'

# Create subplots for each method
fig, axes = plt.subplots(1, 3, figsize=(20, 6))
method_groups = ['Without Failed Attempts', 'With Failed Attempts + Perplexity', 'With Failed Attempts + Thinking']
case_groups = ['5 cases', '10 cases', '15 cases']
case_colors = [color_5_cases, color_10_cases, color_15_cases]

# Find axis ranges - separate for Perplexity and shared for Baseline/Thinking
baseline_thinking_calls = []
perplexity_calls = []
all_coverage = []

for method_group in method_groups:
    for case in case_groups:
        # Get normalization factor based on case group
        case_count = int(case.split()[0])  # Extract number from "5 cases", "10 cases", etc.
        
        for iteration in df.index.get_level_values('Iterations').unique():
            for percentage in df.index.get_level_values('Percentage').unique():
                coverage_val = df.loc[(method_group, iteration, percentage), ('Coverage', case)]
                llm_calls_val = df.loc[(method_group, iteration, percentage), ('LLM Calls', case)]
                all_coverage.append(coverage_val)
                
                # Normalize LLM calls by number of cases
                normalized_calls = llm_calls_val / case_count
                
                if method_group == 'With Failed Attempts + Perplexity':
                    perplexity_calls.append(normalized_calls)
                else:
                    baseline_thinking_calls.append(normalized_calls)

# Axis ranges for baseline and thinking (subplots 0 and 2)
baseline_thinking_min_calls = min(baseline_thinking_calls) - 1
baseline_thinking_max_calls = max(baseline_thinking_calls) + 1

# Axis ranges for perplexity (subplot 1)
perplexity_min_calls = min(perplexity_calls) - 10
perplexity_max_calls = max(perplexity_calls) + 10

# Shared coverage range for all subplots
global_min_coverage = 0  # Keep at 0 for coverage
global_max_coverage = max(all_coverage) + 5  # Add some padding

for idx, method_group in enumerate(method_groups):
    ax = axes[idx]
    
    # Create scatter plots for each case group within this method group
    for case_idx, case in enumerate(case_groups):
        coverage_data = []
        llm_calls_data = []
        
        # Get normalization factor based on case group
        case_count = int(case.split()[0])  # Extract number from "5 cases", "10 cases", etc.
        
        # Iterate through all combinations for this method group and case
        for iteration in df.index.get_level_values('Iterations').unique():
            for percentage in df.index.get_level_values('Percentage').unique():
                coverage_val = df.loc[(method_group, iteration, percentage), ('Coverage', case)]
                llm_calls_val = df.loc[(method_group, iteration, percentage), ('LLM Calls', case)]
                
                coverage_data.append(coverage_val)
                # Normalize LLM calls by number of cases
                llm_calls_data.append(llm_calls_val / case_count)
        
        # Create scatter plot for this case group
        ax.scatter(llm_calls_data, coverage_data, 
                  color=case_colors[case_idx], s=100, 
                  label=case, edgecolors='white', linewidth=0.5)
    
    # Customize each subplot
    ax.set_xlabel('Number of LLM Calls (normalized)', fontsize=14)
    ax.set_ylabel('Line Coverage (%)', fontsize=14)
    
    # Set title based on method group
    if method_group == 'Without Failed Attempts':
        ax.set_title('Baseline', fontsize=16, pad=15)
    elif method_group == 'With Failed Attempts + Perplexity':
        ax.set_title('Baseline + Failed Info + Perplexity', fontsize=16, pad=15)
    else:  # 'With Failed Attempts + Thinking'
        ax.set_title('Baseline + Failed Info + Thinking', fontsize=16, pad=15)
    
    ax.grid(True, linestyle='--', alpha=0.3)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    # Add vertical dashed line at x=1 as reference
    ax.axvline(x=1, color='black', linestyle='--', alpha=0.6, linewidth=1)
    
    # Set axis limits - separate scaling for perplexity subplot
    if idx == 1:  # Perplexity subplot
        ax.set_xlim(perplexity_min_calls, perplexity_max_calls)
    else:  # Baseline and Thinking subplots
        ax.set_xlim(baseline_thinking_min_calls, baseline_thinking_max_calls)
    ax.set_ylim(global_min_coverage, global_max_coverage)
    
    # Add legend only to the last subplot
    if idx == 2:
        ax.legend(fontsize=12, loc='lower right')

# Adjust layout and save
# fig.suptitle('Pareto Plots: Coverage vs LLM Calls by Case Group', fontsize=16, y=1.02)
fig.tight_layout()
plt.savefig('plots/num_calls_vs_coverage_pareto.png', dpi=300, bbox_inches='tight')
plt.close(fig)

print("Pareto plots saved as:")
print("- plots/num_calls_vs_coverage_pareto.png")