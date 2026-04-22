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
* **Deep Learning Framework:** TensorFlow/Keras
* **Deployment:** Joblib (Model Serialization)

## 📊 Data Visualization & Insights

To understand the celestial data, several visualizations were generated using Seaborn and Matplotlib:

* **Class Distribution:** A colorful countplot (using the `magma` palette) shows the frequency of Galaxies, Quasars, and Stars. This highlighted the inherent class imbalance in the SDSS dataset.
* **Feature Relationships:** A Pairplot was utilized to visualize the relationships between different spectral filters (u, g, r, i). This helps identify the "clusters" where different celestial objects reside.
* **Correlation Analysis:** A heatmap was used to identify the strongest relationships between astronomical features, ensuring the most relevant data is fed into the models.

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

![Python](https://img.shields.io/badge/python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54)
![Scikit-Learn](https://img.shields.io/badge/scikit--learn-%23F7931E.svg?style=for-the-badge&logo=scikit-learn&logoColor=white)
![TensorFlow](https://img.shields.io/badge/TensorFlow-%23FF6F00.svg?style=for-the-badge&logo=TensorFlow&logoColor=white)
![Pandas](https://img.shields.io/badge/pandas-%23150458.svg?style=for-the-badge&logo=pandas&logoColor=white)
## ⚙️ Installation & Usage

1. **Clone the Repo:**
   ```bash
   git clone [https://github.com/your-username/predicting_star_quasers_galaxy.git](https://github.com/your-username/predicting_star_quasers_galaxy.git)
