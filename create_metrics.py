import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Set dark theme for modern UI
plt.style.use('dark_background')
sns.set_theme(style="darkgrid", rc={"axes.facecolor": "#1e1e24", "figure.facecolor": "#1e1e24", "text.color": "white", "axes.labelcolor": "white", "xtick.color": "white", "ytick.color": "white"})

# Models and Metrics
models = ['Standard Seq2Seq', 'Pure Rule Engine', 'Sandhi.ai (Neuro-Symbolic)']
accuracies = [62.4, 78.1, 98.6]
oov_handling = [25.0, 40.0, 95.2]

# Plot Setup
fig, ax = plt.subplots(figsize=(8, 5))

# Plotting
x = np.arange(len(models))
width = 0.35

rects1 = ax.bar(x - width/2, accuracies, width, label='Overall Accuracy %', color='#4ECDC4')
rects2 = ax.bar(x + width/2, oov_handling, width, label='OOV Word Handling %', color='#FF6B6B')

# Axis styling
ax.set_ylabel('Performance (%)', fontsize=12, fontweight='bold')
ax.set_title('Sandhi Splitting Engine Benchmarks', fontsize=14, fontweight='bold', pad=15)
ax.set_xticks(x)
ax.set_xticklabels(models, fontsize=10, fontweight='bold')
ax.legend(facecolor='#2d2d37', edgecolor='none')
ax.set_ylim(0, 110)

# Add value labels
def autolabel(rects):
    for rect in rects:
        height = rect.get_height()
        ax.annotate(f'{height}%',
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 3),  # 3 points vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom', fontsize=10, fontweight='bold', color='white')

autolabel(rects1)
autolabel(rects2)

plt.tight_layout()
plt.savefig('images/metrics.png', dpi=300, bbox_inches='tight', transparent=True)
print("Metrics PNG Generated successfully!")
