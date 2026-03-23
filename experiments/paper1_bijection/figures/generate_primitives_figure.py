#!/usr/bin/env python3
"""
Generate the Bayesian Inference Primitives figure for Paper I.
Shows architectural realizability across Transformer, Mamba, and LSTM.
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

# Set up figure
fig, ax = plt.subplots(1, 1, figsize=(12, 8))
ax.set_xlim(0, 10)
ax.set_ylim(0, 8)
ax.axis('off')

# Colors
GREEN = '#4CAF50'  # Full capability
ORANGE = '#FFC107'  # Limited
RED = '#EF5350'    # Cannot realize

# Architecture columns
cols = {'Transformer': 2.5, 'Mamba\n(SSM)': 5, 'LSTM': 7.5}

# Primitive rows (from top to bottom)
rows = {
    'Random-Access\nBinding': 6,
    'Belief\nTransport': 4,
    'Belief\nAccumulation': 2
}

# Task labels (positioned to the right of the figure)
tasks = {
    'Random-Access\nBinding': '(Associative Recall)',
    'Belief\nTransport': '(HMM Filtering)',
    'Belief\nAccumulation': '(Hypothesis Elimination)'
}

# Capability matrix: (color, symbol)
# Transformer: all green with checkmarks
# Mamba: binding=orange/~, transport=green/check, accumulation=green/check
# LSTM: binding=red/X, transport=red/X, accumulation=green/check

capabilities = {
    ('Transformer', 'Random-Access\nBinding'): (GREEN, '\u2713'),
    ('Transformer', 'Belief\nTransport'): (GREEN, '\u2713'),
    ('Transformer', 'Belief\nAccumulation'): (GREEN, '\u2713'),
    ('Mamba\n(SSM)', 'Random-Access\nBinding'): (ORANGE, '~'),
    ('Mamba\n(SSM)', 'Belief\nTransport'): (GREEN, '\u2713'),
    ('Mamba\n(SSM)', 'Belief\nAccumulation'): (GREEN, '\u2713'),
    ('LSTM', 'Random-Access\nBinding'): (RED, '\u2717'),
    ('LSTM', 'Belief\nTransport'): (RED, '\u2717'),
    ('LSTM', 'Belief\nAccumulation'): (GREEN, '\u2713'),
}

# Draw title
ax.text(5, 7.5, 'Bayesian Inference Primitives: Architectural Realizability',
        ha='center', va='center', fontsize=16, fontweight='bold')

# Draw column headers
for arch, x in cols.items():
    ax.text(x, 6.8, arch, ha='center', va='center', fontsize=13, fontweight='bold')

# Draw row labels (primitives) on the left
for prim, y in rows.items():
    ax.text(0.8, y, prim, ha='right', va='center', fontsize=11)

# Draw task labels on the far right (outside the ovals)
for prim, y in rows.items():
    ax.text(9.5, y, tasks[prim], ha='left', va='center', fontsize=10,
            fontstyle='italic', color='#666666')

# Draw ovals and symbols
oval_width = 1.4
oval_height = 0.8

for arch, x in cols.items():
    for prim, y in rows.items():
        color, symbol = capabilities[(arch, prim)]

        # Draw oval
        oval = mpatches.Ellipse((x, y), oval_width, oval_height,
                                 facecolor=color, edgecolor='none', alpha=0.9)
        ax.add_patch(oval)

        # Draw symbol
        symbol_color = 'white'
        ax.text(x, y, symbol, ha='center', va='center',
                fontsize=20, fontweight='bold', color=symbol_color)

# Draw legend at bottom
legend_y = 0.5
legend_items = [
    (GREEN, 'Full capability'),
    (ORANGE, 'Limited (slow/non-scaling)'),
    (RED, 'Cannot realize')
]

legend_x_start = 2
legend_spacing = 2.8

for i, (color, label) in enumerate(legend_items):
    x = legend_x_start + i * legend_spacing
    # Small oval for legend
    oval = mpatches.Ellipse((x, legend_y), 0.4, 0.25,
                             facecolor=color, edgecolor='none', alpha=0.9)
    ax.add_patch(oval)
    ax.text(x + 0.35, legend_y, label, ha='left', va='center', fontsize=10)

plt.tight_layout()
plt.savefig('primitives_figure.png', dpi=150, bbox_inches='tight',
            facecolor='white', edgecolor='none')
plt.savefig('primitives_figure.pdf', bbox_inches='tight',
            facecolor='white', edgecolor='none')
print("Saved primitives_figure.png and primitives_figure.pdf")
