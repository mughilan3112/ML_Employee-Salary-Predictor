# Employee Salary Prediction - Streamlit Web App

A clean and interactive web application for predicting employee salaries using machine learning models.

## 🚀 Features

- **Salary Prediction**: Predict exact salary amounts using Random Forest regression
- **Salary Categories**: Classify employees into Low/Medium/High salary brackets
- **Employee Segmentation**: Group similar employees using K-Means clustering
- **Clean UI**: Modern, responsive interface with intuitive controls
- **Real-time Results**: Instant predictions with detailed insights

## 📊 Models Used

- **Regression Model**: Random Forest Regressor for salary prediction
- **Classification Model**: Decision Tree Classifier for salary categories
- **Clustering Model**: K-Means for employee segmentation

## 🛠️ Installation & Setup

### Prerequisites
- Python 3.8+
- Virtual environment (recommended)

### Installation Steps

1. **Clone or navigate to the project directory**
   ```bash
   cd employee_salary_prediction
   ```

2. **Activate virtual environment** (if using venv)
   ```bash
   # On Windows
   venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Train models** (if not already done)
   ```bash
   python main.py
   ```

5. **Run the web app**
   ```bash
   streamlit run app.py
   ```

## 🎯 Usage

1. Open your browser and go to `http://localhost:8501`
2. Fill in the employee details:
   - Age (22-65)
   - Years of Experience (0-25)
   - Education Level
   - Job Role
   - Work Location
   - Primary Skill
3. Click "🔮 Predict Salary"
4. View the results including:
   - Predicted annual salary
   - Salary category (Low/Medium/High)
   - Employee segment cluster
   - Detailed insights and profile summary

## 📁 Project Structure

```
employee_salary_prediction/
├── app.py                 # Streamlit web application
├── main.py               # Model training and data generation
├── requirements.txt      # Python dependencies
├── data/
│   └── employee_dataset.csv    # Generated dataset
├── models/               # Trained ML models
│   ├── rf_salary_predictor.pkl
│   ├── dt_salary_classifier.pkl
│   └── kmeans_segmentation.pkl
├── outputs/              # Visualizations and results
└── notebooks/            # Jupyter notebooks (if any)
```

## 🔧 Model Training

To retrain the models with new data:

```bash
python main.py
```

This will:
- Generate synthetic employee data
- Create exploratory data analysis visualizations
- Train regression, classification, and clustering models
- Save models to the `models/` directory

## 🎨 UI Features

- **Responsive Design**: Works on desktop and mobile devices
- **Interactive Widgets**: Sliders and dropdowns for easy input
- **Visual Results**: Clean cards displaying predictions
- **Detailed Insights**: Additional context and explanations
- **Professional Styling**: Modern color scheme and typography

## 📈 Model Performance

Based on the training data:
- **Regression R² Score**: ~0.85 (varies with random seed)
- **Classification Accuracy**: ~0.78
- **Clustering**: 3 segments based on experience and salary

## 🤝 Contributing

Feel free to enhance the application by:
- Adding more input features
- Improving the UI/UX design
- Implementing additional ML models
- Adding data validation and error handling
- Creating deployment configurations

## 📄 License

This project is for educational and demonstration purposes.

---

**Built with**: Streamlit, scikit-learn, pandas, matplotlib, seaborn