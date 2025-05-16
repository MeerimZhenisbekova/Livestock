
# Multi-Species Livestock Counting: Differentiation Between Cattle and Sheep

This project applies computer vision and deep learning models to differentiate between cattle and sheep in mixed herds using synthetic image classification data. It was developed as part of a Master's Capstone Project in Data Analytics at CCT College Dublin.

---

## 📁 Project Structure

```
.
├── data_preprocessing.py     # Generates a simulated dataset with labels and environmental variables
├── model_training.py         # Contains accuracy computation logic
├── evaluation.py             # Evaluates model performance and generates visualizations
├── requirements.txt          # Lists dependencies for setting up the environment
├── livestock_model_analysis_plots_labeled.png  # Graphs illustrating findings
```

---

## 🔍 Description

### Data Preprocessing
The synthetic dataset contains 500 labeled samples, each annotated with:
- `True_Label`: Actual species ('Cattle' or 'Sheep')
- `Predicted_Label_{Model}`: Predictions from ResNet, VGG, and AlexNet
- `Lighting_Condition`: Bright, Dim, or Overcast lighting simulation
- `Occlusion_Level`: Low, Medium, or High occlusion scenarios

### Models Simulated
- **ResNet**: Deep residual network with skip connections, good for deeper learning tasks.
- **VGG**: 16-layer convolutional network used for robust feature extraction.
- **AlexNet**: 8-layer architecture known for early CNN success but lower accuracy on modern tasks.

These models were not retrained here but used to simulate prediction behavior under varying conditions.

---

## 📊 Visual Output

`livestock_model_analysis_plots_labeled.png` includes:
- Model accuracy comparison
- ResNet confusion matrix
- Accuracy by lighting condition
- Accuracy by occlusion level

Each subplot is labeled for interpretability.

---

## 🚀 How to Run

### 1. Set up virtual environment (optional but recommended)
```bash
python -m venv livestock-env
source livestock-env/bin/activate  # or `livestock-env\Scripts\activate` on Windows
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Run scripts manually
```bash
python data_preprocessing.py       # To generate synthetic dataset (used inline in notebook)
python model_training.py           # Contains accuracy calculation functions
python evaluation.py               # Outputs labeled graphs and summary visuals
```

---

## 📦 Requirements

See `requirements.txt` for full list. Major dependencies:
- pandas
- numpy
- seaborn
- matplotlib
- scikit-learn

---

## 📄 License

This project is intended for academic research and demonstration purposes only.

---

## 👩‍💻 Author

**Meerim Zhenishbekova**  
MSc Student, Data Analytics  
CCT College Dublin

For more information, refer to the capstone report.

