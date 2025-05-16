
# evaluation.py

import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import confusion_matrix, accuracy_score

def evaluate_and_plot(df, acc_resnet, acc_vgg, acc_alexnet):
    acc_summary = pd.DataFrame({
        'Model': ['ResNet', 'VGG', 'AlexNet'],
        'Accuracy (%)': [round(acc_resnet * 100, 2), round(acc_vgg * 100, 2), round(acc_alexnet * 100, 2)]
    })

    cm = confusion_matrix(df['True_Label'], df['Predicted_Label_ResNet'], labels=['Cattle', 'Sheep'])
    cm_df = pd.DataFrame(cm, columns=['Predicted_Cattle', 'Predicted_Sheep'], index=['Actual_Cattle', 'Actual_Sheep'])

    lighting_accuracy = df.groupby('Lighting_Condition').apply(
        lambda g: accuracy_score(g['True_Label'], g['Predicted_Label_ResNet'])
    ).reset_index(name='ResNet_Accuracy')

    occlusion_accuracy = df.groupby('Occlusion_Level').apply(
        lambda g: accuracy_score(g['True_Label'], g['Predicted_Label_ResNet'])
    ).reset_index(name='ResNet_Accuracy')

    plt.figure(figsize=(14, 10))

    plt.subplot(2, 2, 1)
    sns.barplot(x='Model', y='Accuracy (%)', data=acc_summary, palette='Blues_d')
    plt.title('Model Accuracy Comparison')

    plt.subplot(2, 2, 2)
    sns.heatmap(cm_df, annot=True, fmt='d', cmap='Oranges', cbar=False)
    plt.title('Confusion Matrix: ResNet')

    plt.subplot(2, 2, 3)
    sns.barplot(x='Lighting_Condition', y='ResNet_Accuracy', data=lighting_accuracy, palette='Greens_d')
    plt.title('ResNet Accuracy by Lighting Condition')

    plt.subplot(2, 2, 4)
    sns.barplot(x='Occlusion_Level', y='ResNet_Accuracy', data=occlusion_accuracy, palette='Reds_d')
    plt.title('ResNet Accuracy by Occlusion Level')

    plt.tight_layout()
    plt.savefig('livestock_model_analysis_plots_labeled.png')
