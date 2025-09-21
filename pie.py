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

# Define colors for each component
colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']

# Create pie chart
plt.figure(figsize=(8, 6))
plt.pie(values, labels=labels, colors=colors, autopct='%1.2f%%')
plt.title('Proximate Composition of Plantain')
plt.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.

# Save plot as JPEG
plt.savefig('proximate_composition_plantain_pie.jpg', bbox_inches='tight', dpi=300)

# Show plot
plt.show()
