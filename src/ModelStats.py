import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

# Example data
data = {
    "Split_Type": ["4_parts", "4_parts", "4_parts", "2h_parts", "2h_parts", "2h_parts", "2v_parts", "2v_parts", "2v_parts", "no_splitting", "no_splitting", "no_splitting"],
    "N_Trees": [100, 120, 150, 100, 120, 150, 100, 120, 150, 100, 120, 150],
    "Accuracy": [0.9776, 0.9772, 0.9768, 0.9760, 0.9704, 0.9752, 0.9800, 0.9784, 0.9772, 0.9804, 0.9808, 0.9792]
}


# Convert to a DataFrame for easier plotting
df = pd.DataFrame(data)

"""
plt.figure(figsize=(10, 6))
sns.barplot(x="Split_Type", y="Accuracy", data=df, ci=None)
plt.title("Accuracy by Image Splitting Type")
plt.ylabel("Accuracy")
plt.xlabel("Image Split Type")
plt.show()"""

plt.figure(figsize=(10, 6))
sns.lineplot(x="N_Trees", y="Accuracy", hue="Split_Type", data=df, marker="o")
plt.title("Accuracy vs Number of Trees for Each Split Type")
plt.ylabel("Accuracy")
plt.xlabel("Number of Trees")
plt.legend(title="Split Type")
plt.show()

"""
heatmap_data = df.pivot("Split_Type", "N_Trees", "Accuracy")

plt.figure(figsize=(8, 6))
sns.heatmap(heatmap_data, annot=True, fmt=".2f", cmap="coolwarm")
plt.title("Heatmap of Accuracy by Split Type and Number of Trees")
plt.ylabel("Split Type")
plt.xlabel("Number of Trees")
plt.show()"""
