# predicting_star_quasers_galaxy
# Stellar Classification: Star, Quasar, and Galaxy Prediction

This project implements a machine learning pipeline to classify celestial objects based on spectral data from the **Sloan Digital Sky Survey (SDSS)**. It compares multiple classification algorithms to determine the most accurate model for astronomical identification.

## 🚀 Overview
The backend processes astronomical features such as ultra-violet (u), green (g), red (r), and near-infrared (i, z) filters to categorize objects into three classes:
* **GALAXY**
* **QSO** (Quasar)
* **STAR**

## 🛠️ Tech Stack
* **Language:** Python
* **Data Science:** Pandas, NumPy, Scikit-learn
* **Visualization:** Seaborn, Matplotlib
* **Deep Learning Framework:** TensorFlow/Keras
* **Deployment:** Joblib (Model Serialization)

## 📊 Models Compared
The project evaluates three primary classifiers:
1. **Decision Tree:** Optimized with `max_depth=3` for interpretability.
2. **Logistic Regression:** Used as a baseline linear classifier.
3. **K-Nearest Neighbors (KNN):** Implemented with `n_neighbors=3`.

## 📁 Project Structure
* `predicting_star_quasers_galaxy.ipynb`: The core training logic and EDA.
* `scaler.pkl`: The saved StandardScaler for normalizing input data.
* `dt_model.pkl` / `lr_model.pkl` / `knn_model.pkl`: Pre-trained models.
* `class_names.json`: Mapping for the LabelEncoded classes.

## ⚙️ Installation & Usage

1. **Clone the Repo:**
   ```bash
   git clone [https://github.com/your-username/predicting_star_quasers_galaxy.git](https://github.com/your-username/predicting_star_quasers_galaxy.git)
