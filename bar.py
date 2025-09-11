import pandas as pd
import matplotlib.pyplot as plt

# Sample dataset (replace with your actual data)
data = {
    'Country': ['China', 'India', 'USA', 'Indonesia', 'Pakistan', 'Brazil', 'Nigeria', 'Bangladesh', 'Russia', 'Japan'],
    'Population': [1439323776, 1380004385, 331449281, 273523615, 216565316, 212531000, 202915907, 166303498, 145934029, 127103388]
}

df = pd.DataFrame(data)

# Sort the data by population in descending order
df = df.sort_values(by='Population', ascending=False)

# Create a bar chart
plt.figure(figsize=(10, 6))
plt.bar(df['Country'], df['Population'])
plt.title('Largest Population by Country')
plt.xlabel('Country')
plt.ylabel('Population')
plt.xticks(rotation=90)

# Show the plot
plt.tight_layout()
plt.show()
