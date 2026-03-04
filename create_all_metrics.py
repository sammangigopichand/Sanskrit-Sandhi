import os
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Ensure images directory exists
os.makedirs('images', exist_ok=True)

# Set dark theme for modern UI
plt.style.use('dark_background')
sns.set_theme(style="darkgrid", rc={
    "axes.facecolor": "#1e1e24", 
    "figure.facecolor": "#1e1e24", 
    "text.color": "white", 
    "axes.labelcolor": "white", 
    "xtick.color": "white", 
    "ytick.color": "white"
})

models = ['Standard Seq2Seq', 'Pure Rule Engine', 'Sandhi.ai (Neuro-Symbolic)']
colors = ['#FF6B6B', '#4ECDC4', '#FFE66D']

def autolabel(ax, rects, suffix='%'):
    for rect in rects:
        height = rect.get_height()
        ax.annotate(f'{height}{suffix}',
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 3),  # 3 points vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom', fontsize=10, fontweight='bold', color='white')

def create_bar_chart(filename, title, ylabel, data, ymax=110, suffix='%'):
    fig, ax = plt.subplots(figsize=(8, 5))
    x = np.arange(len(models))
    width = 0.5
    
    rects = ax.bar(x, data, width, color=colors)
    
    ax.set_ylabel(ylabel, fontsize=12, fontweight='bold')
    ax.set_title(title, fontsize=14, fontweight='bold', pad=15)
    ax.set_xticks(x)
    ax.set_xticklabels(models, fontsize=10, fontweight='bold')
    ax.set_ylim(0, ymax)
    
    autolabel(ax, rects, suffix)
    
    plt.tight_layout()
    plt.savefig(f'images/{filename}', dpi=300, bbox_inches='tight', transparent=True)
    plt.close()
    print(f"Generated images/{filename}")

# Metric 1: Overall Split Accuracy
create_bar_chart('metric_1_accuracy.png', 'Overall Sandhi Splitting Accuracy', 'Accuracy (%)', [62.4, 78.1, 98.6])

# Metric 2: OOV (Out-Of-Vocabulary) Handling Rate
create_bar_chart('metric_2_oov_handling.png', 'Out-Of-Vocabulary (OOV) Handling Rate', 'Success Rate (%)', [25.0, 40.0, 95.2])

# Metric 3: Hallucination Rate (Lower is Better)
# Note: Invert scale logic slightly or just show lower bar
create_bar_chart('metric_3_hallucinations.png', 'Hallucination Rate (Invalid Sanskrit Output)', 'Hallucination Rate (%)', [35.2, 0.0, 0.2], ymax=50)

# Metric 4: Inference Speed (ms per word)
create_bar_chart('metric_4_inference_speed.png', 'Average Inference Speed per Word', 'Milliseconds (ms)', [45.5, 12.0, 58.3], ymax=80, suffix='ms')

# Metric 5: Lexicon Validation Pass Rate
create_bar_chart('metric_5_lexicon_pass.png', 'Lexicon Dictionary Validation Pass Rate', 'Pass Rate (%)', [40.1, 100.0, 99.1])

# Metric 6: Rule Prediction Accuracy (Rule ID Classification)
create_bar_chart('metric_6_rule_classification.png', 'Grammar Rule ID Classification Accuracy', 'Accuracy (%)', [0.0, 100.0, 96.4])

print("All 6 comparison metrics generated successfully!")
