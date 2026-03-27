import streamlit as st
import pandas as pd
import joblib
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler

# Set page configuration
st.set_page_config(
    page_title="Employee Salary Predictor",
    page_icon="💼",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .prediction-card {
        background-color: #f0f2f6;
        padding: 1.5rem;
        border-radius: 10px;
        border-left: 5px solid #1f77b4;
        margin: 1rem 0;
    }
    .metric-card {
        background-color: white;
        padding: 1rem;
        border-radius: 8px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        text-align: center;
        margin: 0.5rem;
    }
    .sidebar-header {
        font-size: 1.2rem;
        font-weight: bold;
        color: #1f77b4;
        margin-bottom: 1rem;
    }
</style>
""", unsafe_allow_html=True)

# Load models
@st.cache_resource
def load_models():
    try:
        reg_model = joblib.load('models/rf_salary_predictor.pkl')
        clf_model = joblib.load('models/dt_salary_classifier.pkl')
        kmeans_model = joblib.load('models/kmeans_segmentation.pkl')
        cluster_scaler = joblib.load('models/cluster_scaler.pkl')
        return reg_model, clf_model, kmeans_model, cluster_scaler
    except Exception as e:
        st.error(f"Error loading models: {e}")
        return None, None, None, None

def main():
    # Load models
    reg_model, clf_model, kmeans_model, cluster_scaler = load_models()

    if reg_model is None:
        st.error("Failed to load models. Please ensure the models are trained and saved.")
        return

    # Main header
    st.markdown('<h1 class="main-header">💼 Employee Salary Predictor</h1>', unsafe_allow_html=True)
    st.markdown("""
    <div style='text-align: center; margin-bottom: 2rem;'>
        Predict employee salaries based on professional characteristics using machine learning models.
        Enter the details below and click "Predict Salary" to get insights.
    </div>
    """, unsafe_allow_html=True)

    # Create two columns for layout
    col1, col2 = st.columns([1, 1])

    with col1:
        st.markdown('<h3 class="sidebar-header">📝 Employee Details</h3>', unsafe_allow_html=True)

        # Input widgets
        age = st.slider("Age", min_value=22, max_value=65, value=30,
                       help="Employee's current age")

        experience = st.slider("Years of Experience", min_value=0, max_value=25, value=5,
                              help="Total professional experience in years")

        education = st.selectbox("Education Level",
                                ["Bachelors", "Masters", "PhD"],
                                help="Highest level of education attained")

        job_role = st.selectbox("Job Role",
                               ["Developer", "Data Scientist", "Manager", "Director"],
                               help="Current job position")

        location = st.selectbox("Work Location",
                               ["New York", "San Francisco", "Austin", "Remote"],
                               help="Primary work location")

        skills = st.selectbox("Primary Skill",
                             ["Python", "Java", "SQL", "C++", "AWS"],
                             help="Main technical skill or expertise area")

        # Predict button
        predict_button = st.button("🔮 Predict Salary", type="primary", use_container_width=True)

    with col2:
        st.markdown('<h3 class="sidebar-header">📊 Prediction Results</h3>', unsafe_allow_html=True)

        if predict_button:
            # Create input DataFrame
            input_data = pd.DataFrame({
                'Age': [age],
                'Experience': [experience],
                'Education': [education],
                'Job Role': [job_role],
                'Location': [location],
                'Skills': [skills]
            })

            # Make predictions
            with st.spinner("Analyzing employee profile..."):
                try:
                    # Regression prediction
                    salary_pred = reg_model.predict(input_data)[0]

                    # Classification prediction
                    category_pred = clf_model.predict(input_data)[0]

                    # Clustering prediction (needs Experience and predicted Salary)
                    clust_input = pd.DataFrame({
                        'Experience': [experience],
                        'Salary': [salary_pred]
                    })
                    clust_scaled = cluster_scaler.transform(clust_input)
                    cluster_pred = kmeans_model.predict(clust_scaled)[0]

                    # Display results
                    st.markdown('<div class="prediction-card">', unsafe_allow_html=True)
                    st.markdown("### 🎯 Prediction Results")

                    # Salary prediction with formatting
                    st.markdown(f"""
                    <div style='display: flex; justify-content: space-around; margin: 1rem 0;'>
                        <div class='metric-card'>
                            <h4 style='color: #1f77b4; margin: 0;'>Predicted Salary</h4>
                            <h2 style='color: #2e7d32; margin: 0.5rem 0;'>${salary_pred:,.0f}</h2>
                            <p style='margin: 0; color: #666;'>Annual compensation</p>
                        </div>
                        <div class='metric-card'>
                            <h4 style='color: #1f77b4; margin: 0;'>Salary Category</h4>
                            <h2 style='color: #f57c00; margin: 0.5rem 0;'>{category_pred}</h2>
                            <p style='margin: 0; color: #666;'>Market position</p>
                        </div>
                        <div class='metric-card'>
                            <h4 style='color: #1f77b4; margin: 0;'>Employee Segment</h4>
                            <h2 style='color: #7b1fa2; margin: 0.5rem 0;'>Cluster {cluster_pred}</h2>
                            <p style='margin: 0; color: #666;'>Similar profiles</p>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)

                    # Additional insights
                    st.markdown("### 📈 Insights")

                    # Salary range based on category
                    if category_pred == "Low":
                        range_text = "$50K - $70K"
                        insight = "Entry-level to mid-level position"
                    elif category_pred == "Medium":
                        range_text = "$70K - $120K"
                        insight = "Mid-level professional role"
                    else:  # High
                        range_text = "$120K+"
                        insight = "Senior leadership or specialized role"

                    st.info(f"**Salary Range:** {range_text} | **Profile Insight:** {insight}")

                    # Employee profile summary
                    st.markdown("### 👤 Profile Summary")
                    summary_col1, summary_col2 = st.columns(2)

                    with summary_col1:
                        st.markdown(f"""
                        - **Age:** {age} years
                        - **Experience:** {experience} years
                        - **Education:** {education}
                        - **Role:** {job_role}
                        """)

                    with summary_col2:
                        st.markdown(f"""
                        - **Location:** {location}
                        - **Primary Skill:** {skills}
                        - **Market Value:** {category_pred}
                        - **Segment:** Cluster {cluster_pred}
                        """)

                    st.markdown('</div>', unsafe_allow_html=True)

                except Exception as e:
                    st.error(f"Error making prediction: {e}")
                    st.info("Please check that all models are properly trained and saved.")

        else:
            # Placeholder when no prediction made
            st.info("👈 Enter employee details and click 'Predict Salary' to see results")

            # Show sample visualization
            st.markdown("### 📊 Model Performance Preview")
            st.markdown("Once you make a prediction, you'll see detailed insights and visualizations here.")

    # Footer
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: #666; font-size: 0.8rem;'>
        Built with Streamlit • Powered by Machine Learning • Employee Salary Prediction System
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()