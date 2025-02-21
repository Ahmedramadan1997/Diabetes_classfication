# Diabetes Classification 🩺📊

This project focuses on predicting diabetes by leveraging machine learning techniques and addressing imbalanced data challenges.

---

## 📝 Problem Statement
Diabetes is a critical health issue affecting millions worldwide. Early detection and accurate prediction of diabetes can significantly improve patient outcomes and reduce healthcare costs. The goal of this project is to build a robust machine learning model to predict diabetes based on patient health metrics.

---

## 🔍 Steps Followed

### 1️⃣ **Understanding the Data**
- Explored columns and data types.
- Described numerical and categorical features.
- Cleaned column names for consistency.

### 2️⃣ **Feature Extraction + Exploratory Data Analysis (EDA)**
- **Univariate Analysis**:
  - Analyzed distributions using histograms and distplots.
  - Examined categorical feature frequencies with pie charts and count plots.
- **Bivariate Analysis**:
  - Numerical vs Numerical: Used scatter plots to study relationships.
  - Numerical vs Categorical: Applied box plots, violin plots, and strip plots.
  - Categorical vs Categorical: Visualized comparisons using bar plots and count plots.
- **Multivariate Analysis**:
  - Conducted pair plot analysis for feature relationships.
  - Generated correlation heatmaps for insights.

### 3️⃣ **Pre-Processing**
- **Duplicate Handling**: Identified and removed duplicate records.
- **Train-Test Split**: Divided data into training and testing sets.
- **Missing Values**: Detected and imputed missing values using `SimpleImputer`, `KNNImputer`, and `IterativeImputer`.
- **Outliers**: Addressed outliers using robust statistical techniques.
- **Encoding**:
  - Used `OrdinalEncoder` and `LabelEncoder` for ordinal data.
  - Applied `OneHotEncoder` for nominal data with fewer categories and `BinaryEncoder` for those with more categories.
- **Scaling**: Standardized features using `StandardScaler`, `MinMaxScaler`, and `RobustScaler`.
- **Imbalanced Data**: Balanced data using `SMOTE` for oversampling and `RandomUnderSampler` for undersampling.

### 4️⃣ **Modeling**
- **Baseline Models**: Trained initial models (Logistic Regression, KNN, Decision Trees) to assess performance.
- **Data Balancing Techniques**:
  - Compared models trained with `class_weight='balanced'`.
  - Evaluated models with undersampling and oversampling techniques.
- **Model Comparison**: Analyzed SVM, Random Forest, AdaBoost, and GradientBoosting models.
- **Ensemble Learning**:
  - Experimented with Voting (hard/soft) and Stacking methods.
  - Best Model: Stacked Ensemble with Soft Voting.
- **Threshold Optimization**:
  - Used Precision-Recall Curve to identify a threshold that prioritized recall.
- **Hyperparameter Tuning**:
  - Applied `GridSearchCV` and `RandomizedSearchCV` for optimal settings.

---

## ✅ Results
- **Validation Accuracy**: 97.85%
- **Test Accuracy**: 97.2%
- **Recall**: 40.64%
- **Precision**: 60.64%

---

## 5️⃣ **Model Deployment**
- Saved the best-performing stacked ensemble model using `joblib` for future inference.
- Prepared a pipeline for easy deployment and integration into production systems.

---

## 🛠️ Tools & Libraries
- **Python Libraries**: `numpy`, `pandas`, `seaborn`, `matplotlib`, `plotly`, `scikit-learn`, `imblearn`, `joblib`.
- **Data Visualization**: Histograms, distplots, scatter plots, box plots, violin plots, heatmaps.
- **Machine Learning Models**: Logistic Regression, KNN, SVM, Decision Trees, Random Forests, AdaBoost, GradientBoosting.
- **Ensemble Methods**: Voting, Stacking.
