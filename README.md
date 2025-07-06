# Boston House Prices - ML Learning Project

## Overview

This project implements a machine learning pipeline to predict Boston housing prices using regression techniques. The project focuses on feature engineering, model comparison, and hyperparameter tuning to achieve optimal predictive performance.

## Dataset

The Boston Housing dataset contains information about various features affecting house prices in Boston suburbs, including crime rates, property characteristics, and neighborhood demographics.

This is the _[ Boston House Prices Dataset](https://www.kaggle.com/datasets/fedesoriano/the-boston-houseprice-data)_ on Kaggle, you can download it seperately to use this Notebook.
Naming the file in the main directory "./data/boston.csv"

## Project Structure

### Analysis Pipeline

1. **Data Collection & Investigation**
2. **Data Cleaning pt 1** (handling zero values)
3. **Train/Test Split** - Separate datasets for training and testing
4. **Data Visualization** (pattern recognition, outlier detection)
5. **Data Cleaning pt 2** (filling missing values, adjusting outliers)
6. **Final Data Visualization** (validation of clean data)
7. **Exploratory Analysis** (correlations, feature selection)
8. **Feature Engineering** - optimization and creation of new features
9. **Model Building** - baseline with linear regression
10. **Cross Validation**
11. **Model Iteration** - testing different models and features
12. **Pipeline Implementation** - refactoring cleaning and feature engineering

## Key Learning Outcomes

- **Feature Engineering Impact**: Log and squared transformations successfully reduced data skewness, particularly improving AGE variable distribution from -0.60 to -0.18 skewness
- **Model Performance Comparison**: Random Forest significantly outperformed linear regression, achieving R² of 0.86 vs 0.731
- **Overfitting Detection**: Initial Random Forest showed suspicious R² of 0.970, requiring hyperparameter tuning to achieve realistic performance
- **Transformation Strategy**: Different models benefit from different preprocessing - tree-based models don't always need the same transformations as linear models

## Cross-Validation Analysis

### Model Stability Metrics

- **Mean MAE**: 3.138 ± 0.798
- **Coefficient of Variation**: 0.254 (25.4%)
- **Score Range**: 2.325

### Performance Insights

**Moderate Variance** (CV = 0.254): The model shows some variability across different data splits, suggesting sensitivity to training data composition.

**Notable Range**: The 2.3 point spread indicates model performance varies depending on which houses are used for training.

**Inconsistent Performance**: Standard deviation of 0.8 on a mean of 3.1 shows meaningful variation in predictions.

### Model Stability Assessment

The model is **somewhat unstable** - performance depends on which specific houses are in the training set. This indicates **potential overfitting signals** where high variance suggests the model might be too complex for the dataset size.

## Model Performance Results

### Feature Engineering Impact

Feature engineering with log and squared transformations successfully reduced skewness in data distributions, particularly for the AGE variable. While these transformations improved linear regression performance (R² from 0.727 to 0.731, MAE from 2.81 to 2.58), the biggest gains came from switching to Random Forest.

### Final Model Performance

The tuned Random Forest model achieved the best results with an **R² of 0.870** and **MAE of 2.15**, significantly outperforming linear regression. However, hyperparameter tuning was crucial - the initial Random Forest showed signs of overfitting (R² = 0.970) before being properly regularized.

Linear regression still performed respectably (R² = 0.73, MAE = 2.8), demonstrating that sometimes simpler models can be adequate depending on problem requirements and interpretability needs.

## Technical Stack

- **Python 3.13.x**
- **Pandas** - Data manipulation
- **NumPy** - Numerical computing
- **Matplotlib/Seaborn** - Data visualization
- **Scikit-learn** - Machine learning algorithms
- **Missingno** - Missing data visualization

## Key Takeaways

- Feature transformations work differently for different model types
- Always validate high R² scores to detect overfitting
- Cross-validation is essential for reliable model evaluation
- Hyperparameter tuning can significantly impact model generalization
- Pipelines are excellent for column transformations and reproducible workflows

## Future Improvements

- Implement additional model comparison (Ridge, Lasso, XGBoost)
- Add feature importance analysis
- Include residual plot analysis
- Explore ensemble methods for better stability
