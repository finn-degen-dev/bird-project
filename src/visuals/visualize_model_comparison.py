import matplotlib.pyplot as plt
import numpy as np
import os

# --- Data from your experiments ---
# We take the values you just generated:
# Custom CNN: Memorized training data (99.3%) but failed on test (80.9%)
# ResNet18:   Good training data (93.0%) and generalized well (89.3%)

models = ['Custom CNN\n(From Scratch)', 'ResNet18\n(Transfer Learning)']
train_acc = [99.3, 93.0]  # Accuracy on Training Data
test_acc =  [80.9, 89.3]  # Accuracy on Test Data (Real world performance)

# Setup the bar chart
x = np.arange(len(models))  # label locations
width = 0.35  # width of the bars

fig, ax = plt.subplots(figsize=(10, 6))

# Create bars
rects1 = ax.bar(x - width/2, train_acc, width, label='Training Accuracy', color='#A7C7E7') # Light Blue
rects2 = ax.bar(x + width/2, test_acc, width, label='Test Accuracy', color='#0047AB')    # Dark Blue

# Add textual layout
ax.set_ylabel('Accuracy (%)')
ax.set_title('Comparison: Training from Scratch vs. Transfer Learning')
ax.set_xticks(x)
ax.set_xticklabels(models)
ax.set_ylim(0, 110) # Make room for labels
ax.legend(loc='lower right')

# Add grid for easier reading
ax.yaxis.grid(True, linestyle='--', alpha=0.7)

# Function to add labels on top of bars
def autolabel(rects):
    for rect in rects:
        height = rect.get_height()
        ax.annotate(f'{height}%',
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 3),  # 3 points vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom', fontsize=11, fontweight='bold')

autolabel(rects1)
autolabel(rects2)

# --- Highlight the Overfitting Gap ---
# We draw a red bracket or line to show the gap for Custom CNN
gap_custom = train_acc[0] - test_acc[0]
gap_resnet = train_acc[1] - test_acc[1]

# Annotation for Custom CNN
ax.annotate(f'High Overfitting\nGap: {gap_custom:.1f}%',
            xy=(x[0], (train_acc[0] + test_acc[0])/2),
            xytext=(x[0], 40),
            arrowprops=dict(facecolor='red', shrink=0.05),
            ha='center', color='red', fontweight='bold')

# Annotation for ResNet
ax.annotate(f'Robust Generalization\nGap: {gap_resnet:.1f}%',
            xy=(x[1], (train_acc[1] + test_acc[1])/2),
            xytext=(x[1], 40),
            arrowprops=dict(facecolor='green', shrink=0.05),
            ha='center', color='green', fontweight='bold')

# Save
output_path = "src/visuals/comparison_model_performance.png"
plt.tight_layout()
plt.savefig(output_path, dpi=300)
print(f"Comparison chart saved to {output_path}")
plt.show()
