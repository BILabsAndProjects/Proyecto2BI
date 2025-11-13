import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Get unique values from filtered data
localidades = sorted(dataset['localidad'].unique())
variable = dataset['variable'].iloc[0]
conf_level = dataset['confidence_level'].iloc[0]
year_range = f"{dataset['year_start'].iloc[0]}-{dataset['year_end'].iloc[0]}"

# Create figure with larger size
fig, ax = plt.subplots(figsize=(20, 12))

# Define colors
prussian_blue = '#0B3954'  # Prussian blue for TODAS
orange = '#eb8963'          # Orange for above average
teal = '#087E8B'            # Teal for below average

# Get TODAS mean value
todas_mean = dataset[dataset['localidad'] == 'TODAS']['mean'].values[0] if 'TODAS' in dataset['localidad'].values else None

# Sort localidades to put TODAS first if present
localidades_sorted = sorted(localidades, key=lambda x: (x != 'TODAS', x))

# X positions for each localidad
x_pos = np.arange(len(localidades_sorted))

# Store data for plotting
plot_data = []

# Collect data for each localidad
for i, localidad in enumerate(localidades_sorted):
    loc_data = dataset[dataset['localidad'] == localidad]
    
    if not loc_data.empty:
        mean_val = loc_data['mean'].values[0]
        lower = loc_data['ci_lower'].values[0]
        upper = loc_data['ci_upper'].values[0]
        
        # Determine color
        if localidad == 'TODAS':
            color = prussian_blue
            marker_size = 18
            line_width = 4
            cap_size = 12
            alpha = 1.0
            zorder = 10
        else:
            # Color based on comparison with TODAS mean
            if todas_mean and mean_val > todas_mean:
                color = orange
            else:
                color = teal
            marker_size = 14
            line_width = 3
            cap_size = 8
            alpha = 0.9
            zorder = 5
        
        plot_data.append({
            'x': i,
            'localidad': localidad,
            'mean': mean_val,
            'lower': lower,
            'upper': upper,
            'color': color,
            'marker_size': marker_size,
            'line_width': line_width,
            'cap_size': cap_size,
            'alpha': alpha,
            'zorder': zorder
        })

# Sort by zorder to ensure TODAS is plotted last (on top)
plot_data.sort(key=lambda x: x['zorder'])

# Plot error bars
for data in plot_data:
    ax.errorbar(data['x'], data['mean'], 
               yerr=[[data['mean'] - data['lower']], [data['upper'] - data['mean']]], 
               fmt='o', capsize=data['cap_size'], capthick=data['line_width'],
               color=data['color'], markersize=data['marker_size'],
               alpha=data['alpha'], linewidth=data['line_width'],
               zorder=data['zorder'])

# Add horizontal line at TODAS level for reference
if todas_mean and len(localidades) > 1:
    ax.axhline(y=todas_mean, color=prussian_blue, linestyle='--', alpha=0.3, 
              linewidth=3, label=f'Media TODAS: {todas_mean:.1f}')

# Customize plot with larger fonts
ax.set_xlabel('Localidad', fontsize=32, fontweight='bold')
ax.set_ylabel(f'Concentración de {variable} [μg/m3]', fontsize=26, fontweight='bold')
ax.set_title(f'Medias de Concentración de {variable} por Localidad\n'
            f'Periodo: {year_range}', 
            fontsize=32, fontweight='bold', pad=25)

# Set x-axis with larger font
ax.set_xticks(x_pos)
ax.set_xticklabels(localidades_sorted, rotation=45, ha='right', fontsize=22)

# Highlight TODAS on x-axis if present
for i, label in enumerate(ax.get_xticklabels()):
    if localidades_sorted[i] == 'TODAS':
        label.set_fontweight('bold')
        label.set_fontsize(22)
        label.set_color(prussian_blue)

# Add legend for colors with larger font
if todas_mean and len(localidades) > 1:
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor=prussian_blue, label='TODAS (Media general)'),
        Patch(facecolor=orange, label='Por encima de la media'),
        Patch(facecolor=teal, label='Por debajo de la media')
    ]
    ax.legend(handles=legend_elements, loc='upper right', frameon=True, 
             fancybox=True, shadow=True, fontsize=14)

# Grid
ax.grid(True, alpha=0.3, axis='y', linestyle='--')

# Get extreme CI bounds
min_lower = dataset['ci_lower'].min()
max_upper = dataset['ci_upper'].max()

# Determine reference mean for percentage calculations
if len(localidades) == 1:
    # If only one localidad, use its mean
    reference_mean = dataset['mean'].iloc[0]
else:
    # If multiple localidades, use TODAS mean
    reference_mean = todas_mean if todas_mean else dataset['mean'].mean()

# Calculate percentage differences
lower_pct_diff = ((min_lower - reference_mean) / reference_mean) * 100
upper_pct_diff = ((max_upper - reference_mean) / reference_mean) * 100

# Y-axis limits with some padding
y_min = min_lower - (max_upper - min_lower) * 0.05
y_max = max_upper + (max_upper - min_lower) * 0.05
ax.set_ylim(y_min, y_max)

# Increase y-axis tick font size
ax.tick_params(axis='y', labelsize=12)

# Add annotations for extreme bounds on y-axis
ax.text(-0.05, min_lower, f'({lower_pct_diff:.1f}%)', 
        transform=ax.get_yaxis_transform(),
        ha='right', va='center', fontsize=18, fontweight='bold',
        color='red')

ax.text(-0.05, max_upper, f'(+{upper_pct_diff:.1f}%)', 
        transform=ax.get_yaxis_transform(),
        ha='right', va='center', fontsize=18, fontweight='bold',
        color='red')

# Add horizontal lines at extreme bounds for clarity
ax.axhline(y=min_lower, color='red', linestyle=':', alpha=0.3, linewidth=2)
ax.axhline(y=max_upper, color='red', linestyle=':', alpha=0.3, linewidth=2)

# Add background color
fig.patch.set_facecolor('white')

plt.tight_layout()
plt.show()