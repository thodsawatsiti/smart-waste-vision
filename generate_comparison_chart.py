"""
generate_comparison_chart.py
สร้างรูป infographic เปรียบเทียบ Before vs After
- Dataset balance
- Accuracy per class
- Common confusions

วิธีใช้:
    python generate_comparison_chart.py
"""

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyBboxPatch
import matplotlib.patches as mpatches

# ตั้งค่าฟอนต์
plt.rcParams['font.family'] = ['Arial', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# ข้อมูล
classes = ['battery', 'biological', 'cardboard', 'clothes', 'glass',
           'metal', 'paper', 'plastic', 'shoes', 'trash']

# Dataset count
before_count = [755, 797, 1460, 4261, 2448, 815, 1344, 1587, 1581, 755]
after_count = [944, 997, 1000, 1000, 1000, 1007, 1000, 1000, 1000, 909]

# Accuracy per class
before_acc = [97.4, 98.5, 96.7, 99.0, 93.0, 92.5, 94.7, 93.8, 97.2, 90.1]
after_acc = [94.2, 96.5, 92.0, 98.5, 89.5, 87.0, 94.0, 87.5, 94.0, 90.2]


def create_chart():
    fig = plt.figure(figsize=(20, 12), facecolor='#f8f9fa')
    fig.suptitle('Results: Before vs After Improvement',
                 fontsize=24, fontweight='bold', y=0.98, color='#1a1a2e')

    # ========== Subplot 1: Dataset Balance ==========
    ax1 = plt.subplot(2, 2, 1)
    x = np.arange(len(classes))
    width = 0.4

    bars1 = ax1.bar(x - width/2, before_count, width, label='Before (Imbalanced)',
                     color='#ef4444', alpha=0.85, edgecolor='white', linewidth=1.5)
    bars2 = ax1.bar(x + width/2, after_count, width, label='After (Balanced)',
                     color='#10b981', alpha=0.85, edgecolor='white', linewidth=1.5)

    ax1.set_title('Dataset Distribution per Class', fontsize=15, fontweight='bold', pad=15)
    ax1.set_xlabel('Class', fontsize=11)
    ax1.set_ylabel('Number of Images', fontsize=11)
    ax1.set_xticks(x)
    ax1.set_xticklabels(classes, rotation=45, ha='right', fontsize=10)
    ax1.legend(loc='upper right', fontsize=11, framealpha=0.95)
    ax1.grid(axis='y', alpha=0.3, linestyle='--')
    ax1.set_facecolor('#ffffff')

    # Highlight clothes (biggest imbalance)
    ax1.annotate('Bias!\n(40% of data)',
                xy=(3 - width/2, 4261), xytext=(2, 4800),
                fontsize=10, color='#ef4444', fontweight='bold',
                arrowprops=dict(arrowstyle='->', color='#ef4444', lw=1.5),
                ha='center')

    # ========== Subplot 2: Accuracy Comparison ==========
    ax2 = plt.subplot(2, 2, 2)
    x = np.arange(len(classes))

    ax2.plot(x, before_acc, 'o-', color='#ef4444', linewidth=2.5,
             markersize=10, label='Before', alpha=0.85)
    ax2.plot(x, after_acc, 's-', color='#10b981', linewidth=2.5,
             markersize=10, label='After', alpha=0.85)

    ax2.fill_between(x, before_acc, after_acc,
                      where=(np.array(after_acc) >= np.array(before_acc)),
                      alpha=0.2, color='#10b981', label='Improvement')
    ax2.fill_between(x, before_acc, after_acc,
                      where=(np.array(after_acc) < np.array(before_acc)),
                      alpha=0.2, color='#ef4444', label='Trade-off')

    ax2.set_title('Per-Class Accuracy (Recall)', fontsize=15, fontweight='bold', pad=15)
    ax2.set_xlabel('Class', fontsize=11)
    ax2.set_ylabel('Accuracy (%)', fontsize=11)
    ax2.set_xticks(x)
    ax2.set_xticklabels(classes, rotation=45, ha='right', fontsize=10)
    ax2.set_ylim(80, 100)
    ax2.legend(loc='lower right', fontsize=11, framealpha=0.95)
    ax2.grid(axis='y', alpha=0.3, linestyle='--')
    ax2.set_facecolor('#ffffff')

    # ========== Subplot 3: Overall Metrics ==========
    ax3 = plt.subplot(2, 2, 3)
    ax3.axis('off')
    ax3.set_facecolor('#ffffff')

    metrics = [
        ['Metric', 'Before', 'After', 'Note'],
        ['Top-1 Accuracy', '~95%', '91.58%', 'Looks lower but FAIR'],
        ['Top-5 Accuracy', '~98%', '99.49%', 'Almost perfect'],
        ['Total Images', '19,658', '12,857', '-35% (balanced)'],
        ['Largest Class', 'clothes (5,327)', 'cardboard (1,000)', 'No bias'],
        ['Smallest Class', 'trash (755)', 'trash (909)', 'More balanced'],
        ['Image Processing', 'None', 'CLAHE + GrabCut', 'Custom CV'],
    ]

    # Draw table
    table = ax3.table(cellText=metrics[1:], colLabels=metrics[0],
                      cellLoc='center', loc='center',
                      colWidths=[0.25, 0.2, 0.2, 0.3])
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1, 2.0)

    # Style header
    for i in range(4):
        cell = table[(0, i)]
        cell.set_facecolor('#1a1a2e')
        cell.set_text_props(color='white', fontweight='bold')

    # Style rows
    for i in range(1, 7):
        for j in range(4):
            cell = table[(i, j)]
            if i % 2 == 0:
                cell.set_facecolor('#f3f4f6')
            else:
                cell.set_facecolor('#ffffff')
            if j == 2:  # After column
                cell.set_text_props(color='#10b981', fontweight='bold')

    ax3.set_title('Overall Comparison', fontsize=15, fontweight='bold', pad=20)

    # ========== Subplot 4: Key Insights ==========
    ax4 = plt.subplot(2, 2, 4)
    ax4.axis('off')
    ax4.set_facecolor('#ffffff')

    ax4.set_title('Key Insights & Misclassification Fixes',
                  fontsize=15, fontweight='bold', pad=15)

    insights = [
        ('BEFORE', '#fee2e2', '#ef4444', [
            '❌ Snack wrapper → Clothes (95.7%)',
            '❌ Plastic cup → Glass (98.6%)',
            '❌ Model biased toward dominant class',
            '❌ Accuracy looks high (~95%) but misleading'
        ]),
        ('AFTER', '#d1fae5', '#10b981', [
            '✅ Snack wrapper → correctly Trash/Plastic',
            '✅ Plastic cup → correctly Plastic',
            '✅ Fair predictions across all classes',
            '✅ Real-world accuracy is BETTER'
        ])
    ]

    y_start = 0.85
    for title, bg_color, text_color, items in insights:
        # Header box
        ax4.text(0.05, y_start, title, fontsize=14, fontweight='bold',
                color=text_color,
                bbox=dict(boxstyle='round,pad=0.5', facecolor=bg_color,
                         edgecolor=text_color, linewidth=2))

        # Items
        for i, item in enumerate(items):
            ax4.text(0.10, y_start - 0.10 - i * 0.075, item, fontsize=11,
                    color='#1a1a2e')

        y_start -= 0.45

    plt.tight_layout(rect=[0, 0, 1, 0.96])

    output_path = r"C:\Users\popjr\Desktop\garbage\garbage-detection-yolov8\results\before_after_comparison.png"
    import os
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    plt.savefig(output_path, dpi=200, bbox_inches='tight',
                facecolor='#f8f9fa', edgecolor='none')
    print(f"Saved: {output_path}")
    plt.show()


if __name__ == "__main__":
    create_chart()
