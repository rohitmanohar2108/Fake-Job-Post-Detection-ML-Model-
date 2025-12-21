# Fake Job Posting Detection

Detect fraudulent job postings using a machine learning model trained on job metadata and textual features. This project uses **LightGBM**, **TF-IDF**, and a **Streamlit** frontend for real-time prediction.

You can check out the live project here:  
[Streamlit App – Fake Job Post Detection](https://fake-job-post-detection.streamlit.app/)

---

## Features

- Preprocessing of textual, categorical, and salary data
- TF-IDF vectorization of job descriptions
- One-hot encoding for categorical variables
- Trained LightGBM model with hyperparameter tuning
- Streamlit UI for live job fraud prediction
- Decile analysis and KS-statistics based evaluation
  
---

## Dataset

- **Source:** [Kaggle - Fake Job Postings](https://www.kaggle.com/datasets/shivamb/real-or-fake-fake-jobposting-prediction)
- **Columns include:**
  - `title`, `description`, `requirements`, `company_profile`
  - `employment_type`, `industry`, `function`
  - `location`, `salary_range`
  - `telecommuting`, `has_company_logo`, `has_questions` (binary)

---

## How It Works

### Preprocessing
- Missing value handling
- Extracts and averages salary from `salary_range`
- Parses `location` into `country`, `state`, and `city`
- Combines and cleans text fields for TF-IDF
- One-hot encodes categorical fields
- Creates additional features like number of links in text

### Feature Engineering
- TF-IDF for combined text columns
- Numerical + binary + one-hot + TF-IDF → final feature matrix

### Model
- LightGBM Classifier
- Hyperparameter tuning via `RandomizedSearchCV`
- Evaluation Metrics:
  - Accuracy
  - F1 Score
  - AUC-ROC
  - KS-statistic
  - Gini coefficient

### Streamlit App
- Users enter job details
- Backend generates prediction with fraud probability
- Threshold of `0.4` used to flag fraud

---

## Example Prediction

| Feature            | Example Input                                          |
|--------------------|--------------------------------------------------------|
| Title              | Software Engineer                                      |
| Description        | Looking for backend developer with Django experience   |
| Employment Type    | Full-time                                              |
| Industry           | Information Technology                                 |
| Has Company Logo   | (1)                                                    |
| Has Questions      | (0)                                                    |
| Telecommuting      | (1)                                                    |

**Prediction:** Fraudulent Job (Probability: 0.82)

---

## Running the Project

### Clone the Repository

```bash
git clone https://github.com/yourusername/fake-job-posting-detection.git
cd fake-job-posting-detection
pip install -r requirements.txt
streamlit run app.py
