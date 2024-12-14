# EV-Charging-Session-Duration-Prediction


This project focuses on analyzing electric vehicle (EV) session data and building predictive models to estimate session length and related parameters. The workflow includes data preprocessing, exploratory analysis, outlier detection, clustering, and predictive modeling using various machine learning techniques.

---

## Table of Contents
- [Dataset](#dataset)
- [Objectives](#objectives)
- [Workflow](#workflow)
- [Requirements](#requirements)
- [Key Features](#key-features)
- [Modeling Techniques](#modeling-techniques)
- [Results](#results)
- [How to Run](#how-to-run)
- [Future Enhancements](#future-enhancements)

---

## Dataset
The dataset used in this project includes information about EV charging sessions, weather conditions, and user behavior. Key features include:
- Session length (`SessionLength`)
- Energy delivered (`kWhDelivered`)
- Weather parameters like temperature, humidity, and wind speed
- User and session metadata (e.g., `userID`, `sessionId`)

---

## Objectives
1. Preprocess and clean raw data for analysis.
2. Detect and handle missing values and outliers.
3. Perform clustering to group similar sessions.
4. Build predictive models for estimating session length.
5. Evaluate and compare model performance using key metrics.

---

## Workflow
1. **Data Preprocessing**:
   - Handle missing values and remove irrelevant columns.
   - Standardize numerical features for modeling.
   - Detect and filter outliers using Isolation Forest.

2. **Exploratory Data Analysis**:
   - Analyze correlations between features.
   - Visualize distributions and relationships.

3. **Clustering**:
   - Apply KMeans clustering and determine optimal clusters using the elbow method.

4. **Predictive Modeling**:
   - Train and evaluate models including Random Forest, SVR, XGBoost, and ensemble techniques.

5. **Model Evaluation**:
   - Use metrics like MAE, RMSE, R-squared, and SMAPE.
   - Visualize actual vs. predicted values.

---

## Requirements
The project requires the following Python libraries:

```bash
numpy
pandas
matplotlib
seaborn
scikit-learn
xgboost
keras
```
Install the dependencies using:

```bash
pip install -r requirements.txt
```

---

## Key Features
- **Data Cleaning**: Automated handling of missing values and irrelevant features.
- **Outlier Detection**: Use of Isolation Forest to identify anomalies in session lengths.
- **Clustering**: Group sessions into clusters based on selected features.
- **Predictive Modeling**: Train multiple machine learning models to predict session lengths.
- **Ensemble Learning**: Use stacking and voting regressors to improve accuracy.
- **Visualization**: Comprehensive plots for data analysis and model performance.

---

## Modeling Techniques
1. **Random Forest Regressor**
2. **Support Vector Regression (SVR)**
3. **XGBoost Regressor**
4. **Stacking and Voting Regressors**

---

## Results
Model performance is evaluated using:
- **Mean Absolute Error (MAE)**
- **Root Mean Squared Error (RMSE)**
- **R-squared score**
- **Symmetric Mean Absolute Percentage Error (SMAPE)**

Visualizations include:
- Actual vs. Predicted values (line and scatter plots)
- Regression plots
- Clustering (elbow method)

---

## How to Run
1. Clone the repository:

```bash
git clone https://github.com/yourusername/ev-data-analysis.git
```

2. Navigate to the project directory:

```bash
cd ev-data-analysis
```

3. Ensure dependencies are installed:

```bash
pip install -r requirements.txt
```

4. Update the dataset path in the script:

```python
ev = pd.read_csv(r"your_path_to_dataset/dataset.csv")
```

5. Run the script:

```bash
python main.py
```

---

## Future Enhancements
1. Incorporate additional weather and session-related features.
2. Use advanced hyperparameter tuning methods (e.g., GridSearchCV).
3. Implement deep learning models (e.g., LSTM) for time-series predictions.
4. Develop a web-based interface for visualizing results and predictions.

---

## License
This project is licensed under the MIT License. See the `LICENSE` file for details.

---

## Acknowledgements
Special thanks to the data providers and the open-source community for enabling this project.

