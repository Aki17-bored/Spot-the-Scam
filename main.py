import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

# Load model and vectorizer
model = joblib.load('fake_job_model.pkl')
vectorizer = joblib.load('tfidf_vectorizer.pkl')

# Title
st.title("🛡️ Spot the Scam: Fake Job Detector")

# Upload file
uploaded_file = st.file_uploader("📤 Upload a job listings CSV", type=['csv'])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.write("✅ Uploaded Data Preview:", df.head())

    # Determine which text columns exist in CSV
    text_columns = [col for col in ['title', 'description', 'company_profile', 'requirements', 'benefits'] if col in df.columns]

    # Combine text fields safely
    df['combined_text'] = df[text_columns].fillna('').agg(' '.join, axis=1)

    # Vectorize
    X = vectorizer.transform(df['combined_text'])

    # Predict
    probs = model.predict_proba(X)[:, 1]
    preds = model.predict(X)

    df['fraud_probability'] = probs
    df['prediction'] = preds

    # Display results
    display_cols = ['title', 'location', 'fraud_probability', 'prediction']
    display_cols = [col for col in display_cols if col in df.columns]  # ensure columns exist
    st.write("📋 Prediction Results:", df[display_cols])

    # Pie chart with default labels
    st.subheader("🔍 Real vs Fake Distribution")
    counts = df['prediction'].value_counts()
    labels = ['Real', 'Fake']
    sizes = [counts.get(0, 0), counts.get(1, 0)]  # ensures both classes are shown even if missing
    colors = ['green', 'red']

    fig1, ax1 = plt.subplots()
    ax1.pie(sizes, labels=labels, autopct='%1.1f%%', colors=colors)
    ax1.axis('equal')  # Equal aspect ratio ensures pie chart is circular
    st.pyplot(fig1)

    # Histogram of probabilities
    if 'fraud_probability' in df.columns:
        st.subheader("📊 Fraud Probability Distribution")
        fig2, ax2 = plt.subplots()
        sns.histplot(df['fraud_probability'], bins=20, kde=True, ax=ax2, color='orange')
        st.pyplot(fig2)