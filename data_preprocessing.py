
# data_preprocessing.py

import pandas as pd
import numpy as np

def generate_dataset(seed=42, n_samples=500):
    np.random.seed(seed)
    data = {
        'Image_ID': [f'img_{i}' for i in range(n_samples)],
        'True_Label': np.random.choice(['Cattle', 'Sheep'], size=n_samples),
        'Predicted_Label_ResNet': np.random.choice(['Cattle', 'Sheep'], size=n_samples, p=[0.75, 0.25]),
        'Predicted_Label_VGG': np.random.choice(['Cattle', 'Sheep'], size=n_samples, p=[0.7, 0.3]),
        'Predicted_Label_AlexNet': np.random.choice(['Cattle', 'Sheep'], size=n_samples, p=[0.65, 0.35]),
        'Lighting_Condition': np.random.choice(['Bright', 'Dim', 'Overcast'], size=n_samples),
        'Occlusion_Level': np.random.choice(['Low', 'Medium', 'High'], size=n_samples),
    }
    return pd.DataFrame(data)
