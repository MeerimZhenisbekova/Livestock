
import pandas as pd
import numpy as np
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
import matplotlib.pyplot as plt
import seaborn as sns

# Simulate dataset
np.random.seed(42)
n_samples = 500
data = {
    'Image_ID': [f'img_{i}' for i in range(n_samples)],
    'True_Label': np.random.choice(['Cattle', 'Sheep'], size=n_samples),
    'Predicted_Label_ResNet': np.random.choice(['Cattle', 'Sheep'], size=n_samples, p=[0.75, 0.25]),
    'Predicted_Label_VGG': np.random.choice(['Cattle', 'Sheep'], size=n_samples, p=[0.7, 0.3]),
    'Predicted_Label_AlexNet': np.random.choice(['Cattle', 'Sheep'], size=n_samples, p=[0.65, 0.35]),
    'Lighting_Condition': np.random.choice(['Bright', 'Dim', 'Overcast'], size=n_samples),
    'Occlusion_Level': np.random.choice(['Low', 'Medium', 'High'], size=n_samples),
}

df = pd.DataFrame(data)

# Define accuracy calculation
def compute_accuracy(true, pred):
    return accuracy_score(true, pred)

# Accuracy per model
acc_resnet = compute_accuracy(df['True_Label'], df['Predicted_Label_ResNet'])
acc_vgg = compute_accuracy(df['True_Label'], df['Predicted_Label_VGG'])
acc_alexnet = compute_accuracy(df['True_Label'], df['Predicted_Label_AlexNet'])

# Confusion Matrix for ResNet
cm_resnet = confusion_matrix(df['True_Label'], df['Predicted_Label_ResNet'], labels=['Cattle', 'Sheep'])

# Accuracy by Lighting
lighting_accuracy = df.groupby('Lighting_Condition').apply(
    lambda g: compute_accuracy(g['True_Label'], g['Predicted_Label_ResNet'])
).reset_index(name='ResNet_Accuracy')

# Accuracy by Occlusion
occlusion_accuracy = df.groupby('Occlusion_Level').apply(
    lambda g: compute_accuracy(g['True_Label'], g['Predicted_Label_ResNet'])
).reset_index(name='ResNet_Accuracy')

# Print Summary
print("Model Accuracies:")
print(f"ResNet: {acc_resnet:.2f}")
print(f"VGG: {acc_vgg:.2f}")
print(f"AlexNet: {acc_alexnet:.2f}")

print("
Confusion Matrix for ResNet:")
print(pd.DataFrame(cm_resnet, columns=['Predicted_Cattle', 'Predicted_Sheep'], index=['Actual_Cattle', 'Actual_Sheep']))

print("
Accuracy by Lighting Condition:")
print(lighting_accuracy)

print("
Accuracy by Occlusion Level:")
print(occlusion_accuracy)
