# Employee Salary Prediction & Analysis System

## Project Overview
This represents a comprehensive end-to-end Machine Learning ecosystem aimed at predicting, categorizing, and segmenting employee salaries based on features like Age, Experience, Education, Role, Location, and Skills.

It acts as a full demonstration of the Data Science Lifecycle, involving Data Generation, Exploratory Data Analysis (EDA), Scikit-Learn Pipelines (Data Preprocessing), Regression Modeling, Classification Modeling, and K-Means Clustering. 

## Folder Structure
```text
employee_salary_prediction/
│
├── data/
│   └── employee_dataset.csv       <- Automatically generated realistic dataset
│
├── models/
│   ├── rf_salary_predictor.pkl    <- Random Forest Regressor
│   ├── dt_salary_classifier.pkl   <- Decision Tree Classifier
│   └── kmeans_segmentation.pkl    <- K-Means Clustering Model
│
├── notebooks/                     <- Empty directory intended for Jupyter experiments
│
├── outputs/                       <- Generated visualizations (pngs)
│   ├── salary_distribution.png
│   ├── correlation_heatmap.png
│   ├── feature_importance.png
│   ├── confusion_matrix.png
│   ├── kmeans_elbow_plot.png
│   └── cluster_scatter.png
│
├── main.py                        <- Master execution script
└── README.md                      <- Project documentation (this file)
```

## Running the Application
Ensure you have the required dependencies installed (Pandas, Numpy, Scikit-learn, Matplotlib, Seaborn).

Run the pipeline from your terminal:
```bash
python main.py
```

## System Workflow & Steps
### Step 1: Data Preprocessing
- **Missing Values**: Handled using `SimpleImputer` (median for numerical variables, most frequent for categorical variables).
- **Encoding**: Categorical values (Education, Job Role, Location, Skills) are transformed using `OneHotEncoder`.
- **Scaling**: Numerical features (Age, Experience) are standardized using `StandardScaler`.
- All steps are combined harmoniously inside a `ColumnTransformer` and `Pipeline` to prevent data leakage.

### Step 2: Modeling Paths
1. **Regression Task (Exact Salary Predictor)**
   - Models Evaluated: *Linear Regression*, *Random Forest Regressor*.
   - Output Evaluation: MAE, RMSE, and R² Score metrics.
2. **Classification Task (Salary Bracketer)**
   - Labels created functionally: `Low` (<$70k), `Medium` ($70k-$120k), `High` (>$120k)
   - Models Evaluated: *Logistic Regression*, *Decision Tree Classifier*.
   - Output Evaluation: Accuracy, Classification Report, and Confusion Matrix.
3. **Clustering Task (Employee Segmentation)**
   - Analyzed unlabelled pairs of features like *Experience vs Salary*.
   - Algorithm: *K-Means*
   - Calculated the within-cluster sum of squares (WCSS) via an internal Elbow Plot and established groups mapping identical sub-sections.

### Step 3: Visualization & Outputs
Results ranging from Feature Importance plots (showing which attributes drive salary hikes) to detailed Heatmaps and Distribution plots are automatically dumped into `/outputs/`. A sample dynamic prediction occurs at the end of the script confirming model interactivity.
