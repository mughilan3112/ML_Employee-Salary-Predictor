import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.cluster import KMeans
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import joblib
import warnings

warnings.filterwarnings('ignore')

# Set random seed for reproducibility
np.random.seed(42)

def setup_directories():
    """Create project structure directories if they don't exist"""
    directories = ['data', 'outputs', 'models', 'notebooks']
    for d in directories:
        os.makedirs(d, exist_ok=True)
    print("✓ Project directories verified.")

def generate_data(num_samples=1000):
    """Generates a realistic employee dataset"""
    print("\n--- 1. GENERATING DATASET ---")
    ages = np.random.randint(22, 60, num_samples)
    experience = np.maximum(0, ages - 22 - np.random.randint(0, 5, num_samples))
    
    education_levels = ['Bachelors', 'Masters', 'PhD']
    job_roles = ['Developer', 'Data Scientist', 'Manager', 'Director']
    locations = ['New York', 'San Francisco', 'Austin', 'Remote']
    skills = ['Python', 'Java', 'SQL', 'C++', 'AWS']
    
    data = {
        'Age': ages,
        'Experience': experience,
        'Education': np.random.choice(education_levels, num_samples, p=[0.5, 0.3, 0.2]),
        'Job Role': np.random.choice(job_roles, num_samples),
        'Location': np.random.choice(locations, num_samples),
        'Skills': np.random.choice(skills, num_samples)
    }
    
    df = pd.DataFrame(data)
    
    # Generate Salary based on feature weights with some noise
    base_salary = 50000
    edu_mult = {'Bachelors': 1.0, 'Masters': 1.2, 'PhD': 1.5}
    role_mult = {'Developer': 1.0, 'Data Scientist': 1.1, 'Manager': 1.4, 'Director': 1.8}
    loc_mult = {'New York': 1.2, 'San Francisco': 1.3, 'Austin': 1.0, 'Remote': 0.9}
    
    salaries = []
    for _, row in df.iterrows():
        calc_sal = (base_salary 
                    + (row['Experience'] * 3000)
                    * edu_mult[row['Education']] 
                    * role_mult[row['Job Role']] 
                    * loc_mult[row['Location']])
        # Add random noise
        calc_sal += np.random.normal(0, 5000)
        salaries.append(round(calc_sal))
        
    df['Salary'] = salaries
    
    # Introduce some missing values to demonstrate preprocessing
    idx_to_null = np.random.choice(df.index, size=50, replace=False)
    df.loc[idx_to_null, 'Experience'] = np.nan
    
    df.to_csv('data/employee_dataset.csv', index=False)
    print(f"✓ Dataset generated and saved to 'data/employee_dataset.csv' with {num_samples} records.")
    return df

def visualize_eda(df):
    """Creates exploratory data analysis charts"""
    print("\n--- 2. GENERATING EDA VISUALIZATIONS ---")
    
    # Set seaborn style
    sns.set_theme(style="whitegrid")
    
    # 1. Salary Distribution
    plt.figure(figsize=(8, 5))
    sns.histplot(df['Salary'], bins=30, kde=True, color='blue')
    plt.title('Salary Distribution')
    plt.xlabel('Salary ($)')
    plt.savefig('outputs/salary_distribution.png')
    plt.close()
    
    # 2. Correlation Heatmap (only numeric features)
    numeric_df = df.select_dtypes(include=[np.number])
    plt.figure(figsize=(8, 6))
    sns.heatmap(numeric_df.corr(), annot=True, cmap='coolwarm', fmt=".2f", vmin=-1, vmax=1)
    plt.title('Feature Correlation Heatmap')
    plt.tight_layout()
    plt.savefig('outputs/correlation_heatmap.png')
    plt.close()
    
    print("✓ EDA Visualizations saved to 'outputs/' directory.")

def preprocess_and_train_regression(df):
    """Trains a regression model to predict exact salaries"""
    print("\n--- 3A. REGRESSION (SALARY PREDICTION) ---")
    
    X = df.drop('Salary', axis=1)
    y = df['Salary']
    
    categorical_cols = ['Education', 'Job Role', 'Location', 'Skills']
    numeric_cols = ['Age', 'Experience']
    
    # Preprocessing pipelines for both numeric and categorical data
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')), # Handling missing values
        ('scaler', StandardScaler())])                 # Feature scaling
        
    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')), 
        ('onehot', OneHotEncoder(handle_unknown='ignore'))]) # One-Hot Encoding
        
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_cols),
            ('cat', categorical_transformer, categorical_cols)])
            
    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    print(f"✓ Data split: Train={X_train.shape[0]}, Test={X_test.shape[0]}")
    
    # Model 1: Linear Regression
    lr_pipeline = Pipeline(steps=[('preprocessor', preprocessor), ('regressor', LinearRegression())])
    lr_pipeline.fit(X_train, y_train)
    lr_preds = lr_pipeline.predict(X_test)
    
    print("\n> Linear Regression Evaluation:")
    print(f"  MAE:  ${mean_absolute_error(y_test, lr_preds):,.2f}")
    print(f"  RMSE: ${np.sqrt(mean_squared_error(y_test, lr_preds)):,.2f}")
    print(f"  R²:   {r2_score(y_test, lr_preds):.4f}")
    
    # Model 2: Random Forest Regressor
    rf_pipeline = Pipeline(steps=[('preprocessor', preprocessor), 
                                  ('regressor', RandomForestRegressor(n_estimators=100, random_state=42))])
    rf_pipeline.fit(X_train, y_train)
    rf_preds = rf_pipeline.predict(X_test)
    
    print("\n> Random Forest Regressor Evaluation:")
    print(f"  MAE:  ${mean_absolute_error(y_test, rf_preds):,.2f}")
    print(f"  RMSE: ${np.sqrt(mean_squared_error(y_test, rf_preds)):,.2f}")
    print(f"  R²:   {r2_score(y_test, rf_preds):.4f}")
    
    # Save best model
    joblib.dump(rf_pipeline, 'models/rf_salary_predictor.pkl')
    print("✓ Best Regression model (Random Forest) saved to 'models/'")
    
    # Feature Importance Visualization
    plt.figure(figsize=(10, 6))
    
    # Extract feature names after One-Hot Encoding
    ohe_feature_names = rf_pipeline.named_steps['preprocessor'] \
        .transformers_[1][1].named_steps['onehot'].get_feature_names_out(categorical_cols)
    feature_names = numeric_cols + list(ohe_feature_names)
    
    importances = rf_pipeline.named_steps['regressor'].feature_importances_
    indices = np.argsort(importances)[-10:] # Top 10 features
    
    plt.barh(range(len(indices)), importances[indices], align='center', color='teal')
    plt.yticks(range(len(indices)), [feature_names[i] for i in indices])
    plt.xlabel('Relative Importance')
    plt.title('Top 10 Feature Importances (Random Forest)')
    plt.tight_layout()
    plt.savefig('outputs/feature_importance.png')
    plt.close()
    
    return rf_pipeline, preprocessor

def train_classification(df):
    """Trains a classification model to predict salary brackets"""
    print("\n--- 3B. CLASSIFICATION (SALARY CATEGORY) ---")
    
    # Convert Continuous Salary into Categories
    def categorize_salary(sal):
        if sal < 70000: return 'Low'
        elif sal <= 120000: return 'Medium'
        else: return 'High'
        
    df['Salary_Category'] = df['Salary'].apply(categorize_salary)
    
    X = df.drop(['Salary', 'Salary_Category'], axis=1)
    y = df['Salary_Category']
    
    categorical_cols = ['Education', 'Job Role', 'Location', 'Skills']
    numeric_cols = ['Age', 'Experience']
    
    # Preprocessor (same principles)
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', Pipeline(steps=[('imputer', SimpleImputer(strategy='median')), ('scaler', StandardScaler())]), numeric_cols),
            ('cat', Pipeline(steps=[('imputer', SimpleImputer(strategy='most_frequent')), ('onehot', OneHotEncoder(handle_unknown='ignore'))]), categorical_cols)])
            
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Model 1: Logistic Regression
    clf_log = Pipeline(steps=[('preprocessor', preprocessor),
                              ('classifier', LogisticRegression(max_iter=1000, random_state=42))])
    clf_log.fit(X_train, y_train)
    log_preds = clf_log.predict(X_test)
    
    print("\n> Logistic Regression Results:")
    print(f"  Accuracy: {accuracy_score(y_test, log_preds):.4f}")
    
    # Model 2: Decision Tree
    clf_dt = Pipeline(steps=[('preprocessor', preprocessor),
                             ('classifier', DecisionTreeClassifier(random_state=42, max_depth=8))])
    clf_dt.fit(X_train, y_train)
    dt_preds = clf_dt.predict(X_test)
    
    print("\n> Decision Tree Results:")
    print(f"  Accuracy: {accuracy_score(y_test, dt_preds):.4f}")
    
    print("\nClassification Report (Decision Tree):")
    print(classification_report(y_test, dt_preds))
    
    # Confusion Matrix Visualization
    labels = ['Low', 'Medium', 'High']
    cm = confusion_matrix(y_test, dt_preds, labels=labels)
    plt.figure(figsize=(6, 4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=labels, yticklabels=labels)
    plt.ylabel('Actual Category')
    plt.xlabel('Predicted Category')
    plt.title('Confusion Matrix - Decision Tree')
    plt.savefig('outputs/confusion_matrix.png')
    plt.close()
    
    joblib.dump(clf_dt, 'models/dt_salary_classifier.pkl')
    print("✓ Classification model (Decision Tree) saved.")
    return clf_dt

def perform_clustering(df):
    """Performs K-Means clustering to segment employees"""
    print("\n--- 3C. CLUSTERING (EMPLOYEE SEGMENTATION) ---")
    
    # Using Experience and Salary for 2D visualization
    df_cluster = df[['Experience', 'Salary']].copy()
    
    # Handling missing values
    df_cluster['Experience'] = df_cluster['Experience'].fillna(df_cluster['Experience'].median())
    
    # Scaling is crucial for distance-based algorithms like K-Means
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(df_cluster)
    
    # Elbow Method (Optional block: we assume K=3 for demonstration, but here is the code calculating it)
    wcss = []
    for i in range(1, 11):
        kmeans_temp = KMeans(n_clusters=i, init='k-means++', random_state=42, n_init=10)
        kmeans_temp.fit(X_scaled)
        wcss.append(kmeans_temp.inertia_)
        
    plt.figure(figsize=(8, 5))
    plt.plot(range(1, 11), wcss, marker='o', linestyle='--', color='purple')
    plt.title('Elbow Method to Determine Optimal K')
    plt.xlabel('Number of Clusters (K)')
    plt.ylabel('Within-Cluster Sum of Squares (WCSS)')
    plt.savefig('outputs/kmeans_elbow_plot.png')
    plt.close()
    
    # Fit final clustering algorithm
    k = 3
    kmeans = KMeans(n_clusters=k, init='k-means++', random_state=42, n_init=10)
    clusters = kmeans.fit_predict(X_scaled)
    df['Cluster'] = clusters
    
    # Cluster Visualization
    plt.figure(figsize=(8, 6))
    sns.scatterplot(x='Experience', y='Salary', hue='Cluster', palette='Set1', data=df, style='Cluster', s=100)
    plt.title(f'Employee Segmentation mapped to Clusters (K={k})')
    plt.savefig('outputs/cluster_scatter.png')
    plt.close()
    
    joblib.dump(kmeans, 'models/kmeans_segmentation.pkl')
    joblib.dump(scaler, 'models/cluster_scaler.pkl')
    print(f"✓ Clustering completed. Elbow plot and Scatter plot saved.")
    return kmeans, scaler

def predict_single_sample(reg_model, clf_model, kmeans_model, cluster_scaler):
    """Outputs a sample prediction from end to end"""
    print("\n" + "="*50)
    print("--- 4. END-TO-END SYSTEM SAMPLE PREDICTION ---")
    
    # Create a dummy profile
    sample_data = pd.DataFrame({
        'Age': [28],
        'Experience': [4.0],
        'Education': ['Masters'],
        'Job Role': ['Data Scientist'],
        'Location': ['San Francisco'],
        'Skills': ['Python']
    })
    
    print("\n>>> Input Employee Profile:")
    print(sample_data.to_markdown(index=False))
    
    # Results
    pred_salary = reg_model.predict(sample_data)[0]
    pred_category = clf_model.predict(sample_data)[0]
    
    # For clustering predict shape must match fitting features (Experience, Salary)
    clust_input = pd.DataFrame({'Experience': [sample_data['Experience'].values[0]], 'Salary': [pred_salary]})
    clust_scaled = cluster_scaler.transform(clust_input)
    pred_cluster = kmeans_model.predict(clust_scaled)[0]
    
    print("\n>>> SYSTEM OUTPUT:")
    print(f"  [Regression]     Predicted Salary:   ${pred_salary:,.2f}")
    print(f"  [Classification] Predicted Category: {pred_category}")
    print(f"  [Clustering]     Assigned Segment:   Cluster {pred_cluster}")
    print("="*50 + "\n")

def main():
    print("Starting Employee Salary Prediction & Analysis System pipeline...\n")
    setup_directories()
    
    # 1. Generate & load
    df = generate_data(num_samples=1500)
    
    # 2. Exploratory Data Analysis
    visualize_eda(df)
    
    # 3. Modeling
    reg_model, preprocessor = preprocess_and_train_regression(df.copy())
    clf_model = train_classification(df.copy())
    kmeans_model, cluster_scaler = perform_clustering(df.copy())
    
    # 4. Output Example Predictions
    predict_single_sample(reg_model, clf_model, kmeans_model, cluster_scaler)
    
    print("Pipeline executed successfully! Check the outputs/ and models/ folders.")

if __name__ == "__main__":
    main()
