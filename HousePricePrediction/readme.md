# Task 2: House Price Prediction Pipeline


## 📌 Project Overview
This project implements a comprehensive Machine Learning pipeline to predict house prices. It transitions from basic linear modeling to a robust workflow involving feature scaling, data auditing, and multi-metric evaluation. This task serves as a foundation for understanding regression analysis and feature importance in predictive modeling.

## 🚀 Key Features
* **Automated Data Quality Audit**: A custom Python function to verify dataset dimensions, feature sets, and missing values before training.
* **Feature Scaling**: Implementation of `StandardScaler` to normalize numerical features, ensuring that variables with different units (e.g., Area vs. Rooms) are treated equitably by the model.
* **Linear Regression Baseline**: Established a predictive baseline for continuous house price values.
* **Advanced Evaluation Metrics**: Model performance is assessed using:
    * **Mean Absolute Error (MAE)**: To provide an interpretable average error in price.
    * **R-squared (R²)**: To measure the variance captured by the model.
* **Prediction Visualization**: Scatter plot analysis of "Actual vs. Predicted" prices to visually diagnose model accuracy.

## 🛠️ Tech Stack
* **Language**: Python 3.x
* **Data Processing**: Pandas, NumPy
* **Machine Learning**: Scikit-Learn
* **Visualization**: Seaborn, Matplotlib

## 📊 Summary of Findings
- **Scaling Impact**: Normalizing the data prevented large-scale features from dominating the regression coefficients.
- **Model Accuracy**: The baseline model provides a reliable R² score, though further improvements could include Decision Tree or Random Forest regressors for non-linear patterns.
- **Data Integrity**: The automated audit ensures that no null values interfere with the training pipeline.

