import seaborn as sns
import matplotlib.pyplot as plt

# Proximate composition of plantain (example values)
composition = {
    'Moisture': 95.62,
    'Carbohydrates': 2.34,
    'Protein': 1.63,
    'Fat': 0.25,
    'Fiber': 0.25,
    'Ash': 0.15
}

# Extract labels and values
labels = list(composition.keys())
values = list(composition.values())

# Create bar chart
sns.set()
plt.figure(figsize=(8, 6))
sns.barplot(x=labels, y=values)
plt.xlabel('Component')
plt.ylabel('Percentage (%)')
plt.title('Proximate Composition of Plantain')

# Save plot as JPEG
plt.savefig('proximate_composition_plantain.jpg', bbox_inches='tight', dpi=300)

# Show plot
plt.show()
