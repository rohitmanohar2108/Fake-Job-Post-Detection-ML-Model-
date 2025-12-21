import streamlit as st
import pandas as pd
import numpy as np
import re
from joblib import load
from scipy.sparse import csr_matrix, hstack
import os

# Page configuration
st.set_page_config(
    page_title="Job Posting Fraud Detection",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    /* Reduce top padding and margins */
    .main .block-container {
        padding-top: 0rem;
        padding-bottom: 0rem;
        margin-top: 0rem;
    }
    
    /* Remove default streamlit header spacing */
    .stApp > header {
        background-color: transparent;
    }
    
    /* Reduce space in main content area */
    .main > div {
        padding-top: 0.5rem;
    }
    
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .fraud-high {
        background-color: #ffebee;
        border-left: 5px solid #f44336;
        padding: 1rem;
        border-radius: 5px;
        margin: 1rem 0;
        color: #c62828;
    }
    .fraud-low {
        background-color: #e8f5e8;
        border-left: 5px solid #4caf50;
        padding: 1rem;
        border-radius: 5px;
        margin: 1rem 0;
        color: #2e7d32;
    }
    .fraud-medium {
        background-color: #fff3e0;
        border-left: 5px solid #ff9800;
        padding: 1rem;
        border-radius: 5px;
        margin: 1rem 0;
        color: #ef6c00;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_model():
    """Load the trained model and components"""
    try:
        model_data = load('model_data.joblib')
        return model_data['model'], model_data['features'], model_data['tfidf_vectorizer']
    except FileNotFoundError:
        st.error("Model file 'artifacts/model_data.joblib' not found. Please ensure the model is trained and saved.")
        return None, None, None
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        return None, None, None

def process_salary_range(salary_range):
    """Process salary range exactly as done during training"""
    if pd.isna(salary_range) or salary_range == "" or not str(salary_range).strip():
        salary_range = "0-0"
    
    # Split salary range
    salary_parts = str(salary_range).split('-')
    if len(salary_parts) != 2:
        return 0  # Default to 0 if format is incorrect
    
    # Clean strings (remove non-numeric characters)
    salary_min_str = re.sub(r'[^0-9]', '', salary_parts[0])
    salary_max_str = re.sub(r'[^0-9]', '', salary_parts[1])
    
    # Convert to numeric safely
    try:
        salary_min = float(salary_min_str) if salary_min_str else 0
        salary_max = float(salary_max_str) if salary_max_str else 0
    except ValueError:
        return 0
    
    # Compute average salary
    salary_avg = (salary_min + salary_max) / 2
    return salary_avg if not pd.isna(salary_avg) else 0

def preprocess_text(text):
    """Preprocess text exactly as done during training"""
    if pd.isna(text) or text == "":
        text = "missing"
    text = str(text).lower()
    text = re.sub(r'[^a-z\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def create_feature_vector(user_inputs, feature_names, tfidf_vectorizer):
    """Create feature vector from user inputs"""
    
    # Initialize feature vector with zeros
    feature_dict = {feature: 0 for feature in feature_names}
    
    # Process text fields
    text_fields = ['company_profile', 'description', 'requirements', 'title', 'benefits']
    processed_texts = []
    
    for field in text_fields:
        processed_text = preprocess_text(user_inputs.get(field, ''))
        processed_texts.append(processed_text)
    
    # Combine all text
    all_text = ' '.join(processed_texts)
    
    # Transform text using TF-IDF
    tfidf_features = tfidf_vectorizer.transform([all_text])
    
    # Set non-text features
    # Boolean features
    if 'telecommuting' in feature_dict:
        feature_dict['telecommuting'] = int(user_inputs.get('telecommuting', False))
    if 'has_company_logo' in feature_dict:
        feature_dict['has_company_logo'] = int(user_inputs.get('has_company_logo', False))
    if 'has_questions' in feature_dict:
        feature_dict['has_questions'] = int(user_inputs.get('has_questions', False))
    
    # URL count (set to 0 as we're not processing URLs in this simple interface)
    if 'url_count' in feature_dict:
        feature_dict['url_count'] = 0
    
    # Process salary range
    salary_avg = process_salary_range(user_inputs.get('salary_range', ''))
    if 'salary_avg' in feature_dict:
        feature_dict['salary_avg'] = salary_avg
    
    # One-hot encoded features
    employment_type = user_inputs.get('employment_type', '')
    if employment_type and employment_type != 'Select...':
        employment_key = f'employment_type_{employment_type.replace(" ", "_").replace("-", "_").lower()}'
        if employment_key in feature_dict:
            feature_dict[employment_key] = 1
    
    required_experience = user_inputs.get('required_experience', '')
    if required_experience and required_experience != 'Select...':
        experience_key = f'required_experience_{required_experience.replace(" ", "_").replace("-", "_").lower()}'
        if experience_key in feature_dict:
            feature_dict[experience_key] = 1
    
    required_education = user_inputs.get('required_education', '')
    if required_education and required_education != 'Select...':
        education_key = f'required_education_{required_education.replace(" ", "_").replace("-", "_").replace("\'", "").lower()}'
        if education_key in feature_dict:
            feature_dict[education_key] = 1
    
    # Create numeric feature vector (excluding TF-IDF features)
    tfidf_feature_names = tfidf_vectorizer.get_feature_names_out()
    numeric_features = [name for name in feature_names if name not in tfidf_feature_names]
    
    numeric_vector = np.array([feature_dict[name] for name in numeric_features]).reshape(1, -1)
    numeric_sparse = csr_matrix(numeric_vector)
    
    # Combine numeric and text features
    combined_features = hstack([numeric_sparse, tfidf_features])
    
    return combined_features

def main():
    st.markdown('<h1 class="main-header">Job Posting Fraud Detection</h1>', unsafe_allow_html=True)
    
    # Load model
    model, feature_names, tfidf_vectorizer = load_model()
    
    if model is None:
        return
    
    st.markdown("### Enter job posting details to check for potential fraud")
    
    # Create two columns for input
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Job Information")
        
        title = st.text_input("Job Title", placeholder="e.g., Software Engineer")
        
        company_profile = st.text_area(
            "Company Profile", 
            height=100,
            placeholder="Brief description of the company..."
        )
        
        description = st.text_area(
            "Job Description", 
            height=150,
            placeholder="Detailed job description..."
        )
        
        requirements = st.text_area(
            "Job Requirements", 
            height=100,
            placeholder="Required skills and qualifications..."
        )
        
        benefits = st.text_area(
            "Benefits", 
            height=80,
            placeholder="Benefits offered..."
        )
        
        # Salary Range
        st.subheader("Compensation")
        salary_range = st.text_input(
            "Salary Range", 
            placeholder="e.g., 50000-80000 or 50k-80k"
        )
    
    with col2:
        st.subheader("Job Details")
        
        # Employment type
        employment_types = ['Select...', 'Full-time', 'Part-time', 'Contract', 'Temporary', 'Other']
        employment_type = st.selectbox("Employment Type", employment_types)
        
        # Required experience
        experience_levels = ['Select...', 'Entry level', 'Mid level', 'Senior level', 'Executive', 'Not Applicable']
        required_experience = st.selectbox("Required Experience", experience_levels)
        
        # Required education
        education_levels = ['Select...', 'High School', 'Some College', 'Bachelor Degree', 'Master Degree', 'Doctorate', 'Professional', 'Certification', 'Vocational']
        required_education = st.selectbox("Required Education", education_levels)
        
        # Location fields
        st.subheader("Location & Category")
        col2a, col2b = st.columns(2)
        
        with col2a:
            country = st.text_input("Country", placeholder="e.g., US")
            state = st.text_input("State/Province", placeholder="e.g., California")
            city = st.text_input("City", placeholder="e.g., San Francisco")
        
        with col2b:
            industry = st.text_input("Industry", placeholder="e.g., Technology")
            function = st.text_input("Function", placeholder="e.g., Engineering")
            department = st.text_input("Department", placeholder="e.g., Software Development")
        
        # Boolean features
        st.subheader("Additional Features")
        telecommuting = st.checkbox("Remote Work Available")
        has_company_logo = st.checkbox("Company Logo Present")
        has_questions = st.checkbox("Screening Questions Present")
    
    # Prediction button
    if st.button("Analyze Job Posting", type="primary", use_container_width=True):
        
        # Collect user inputs
        user_inputs = {
            'title': title,
            'company_profile': company_profile,
            'description': description,
            'requirements': requirements,
            'benefits': benefits,
            'salary_range': salary_range,
            'employment_type': employment_type,
            'required_experience': required_experience,
            'required_education': required_education,
            'industry': industry,
            'function': function,
            'department': department,
            'country': country,
            'state': state,
            'city': city,
            'telecommuting': telecommuting,
            'has_company_logo': has_company_logo,
            'has_questions': has_questions
        }
        
        try:
            # Create feature vector
            feature_vector = create_feature_vector(user_inputs, feature_names, tfidf_vectorizer)
            
            # Make prediction
            fraud_probability = model.predict_proba(feature_vector)[0, 1]
            is_fraudulent = fraud_probability > 0.4
            
            # Display results
            st.markdown("---")
            st.subheader("Analysis Results")
            
            # Create columns for results
            result_col1, result_col2 = st.columns(2)
            
            with result_col1:
                st.metric("Fraud Probability", f"{fraud_probability:.1%}")
                
                # Progress bar for probability
                st.progress(fraud_probability)
            
            with result_col2:
                if fraud_probability >= 0.7:
                    st.markdown(f'''
                    <div class="fraud-high">
                        <h3>HIGH RISK</h3>
                        <p><strong>Classification:</strong> Likely Fraudulent</p>
                        <p><strong>Recommendation:</strong> Exercise extreme caution. This job posting shows strong indicators of fraud.</p>
                    </div>
                    ''', unsafe_allow_html=True)
                elif fraud_probability >= 0.4:
                    st.markdown(f'''
                    <div class="fraud-medium">
                        <h3>MEDIUM RISK</h3>
                        <p><strong>Classification:</strong> Potentially Fraudulent</p>
                        <p><strong>Recommendation:</strong> Review carefully and verify company details before proceeding.</p>
                    </div>
                    ''', unsafe_allow_html=True)
                else:
                    st.markdown(f'''
                    <div class="fraud-low">
                        <h3>LOW RISK</h3>
                        <p><strong>Classification:</strong> Likely Legitimate</p>
                        <p><strong>Recommendation:</strong> This job posting appears to be legitimate, but always exercise normal caution.</p>
                    </div>
                    ''', unsafe_allow_html=True)
            
            # Risk factors
            st.subheader("Risk Analysis")
            
            risk_factors = []
            if not title.strip():
                risk_factors.append("Missing job title")
            
            if not company_profile.strip():
                risk_factors.append("Missing company profile")
            if not description.strip():
                risk_factors.append("Missing job description")
            if employment_type == 'Select...':
                risk_factors.append("Employment type not specified")
            if not country.strip():
                risk_factors.append("Missing location information")
            if not salary_range.strip():
                risk_factors.append("Missing salary information")
            
            positive_factors = []
            if has_company_logo:
                positive_factors.append("Company logo present")
            if has_questions:
                positive_factors.append("Screening questions present")
            if len(description) > 100:
                positive_factors.append("Detailed job description")
            if benefits.strip():
                positive_factors.append("Benefits information provided")
            if salary_range.strip() and salary_range != "0-0":
                positive_factors.append("Salary range specified")
            
            col_risk, col_positive = st.columns(2)
            
            with col_risk:
                if risk_factors:
                    st.markdown("**Potential Risk Factors:**")
                    for factor in risk_factors:
                        st.markdown(factor)
                else:
                    st.markdown("**No obvious risk factors detected**")
            
            with col_positive:
                if positive_factors:
                    st.markdown("**Positive Indicators:**")
                    for factor in positive_factors:
                        st.markdown(factor)
                else:
                    st.markdown("**Consider adding more details to improve assessment**")
        
        except Exception as e:
            st.error(f"Error during prediction: {str(e)}")
            st.info("Please check that all required fields are filled correctly.")

    # Sidebar with information
    with st.sidebar:
        st.markdown("## About This Tool")
        st.markdown("""
        This fraud detection system analyzes job postings to identify potential scams using machine learning.
        
        **How it works:**
        - Analyzes text content using LIghtGBM techniques
        - Considers job posting structure and completeness
        - Uses patterns learned from known fraudulent postings
        
        **Tips for users:**
        - Provide as much detail as possible
        - Be wary of postings with unusually high fraud probability
        - Always verify company information independently
        
        **Threshold:** Jobs with >40% fraud probability are flagged as potentially fraudulent.
        """)
        
        st.markdown("---")
        st.markdown("**Privacy Note:** All data is processed locally and not stored.")

if __name__ == "__main__":
    main()
