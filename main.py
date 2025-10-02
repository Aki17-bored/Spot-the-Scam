import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt

# Load model and vectorizer
model = joblib.load('fake_job_model.pkl')
vectorizer = joblib.load('tfidf_vectorizer.pkl')

st.set_page_config(page_title="🛡️ Spot the Scam", layout="wide")
st.title("🛡️ Spot the Scam: Fake Job Detector")

def predict_jobs(file):
    try:
        # Read CSV
        df = pd.read_csv(file)

        # Determine which text columns exist
        text_columns = [col for col in ['title', 'description', 'company_profile', 'requirements', 'benefits'] if col in df.columns]

        if not text_columns:
            st.error("No text columns found in the CSV!")
            return None

        # Combine text fields
        df['combined_text'] = df[text_columns].fillna('').agg(' '.join, axis=1)

        # Vectorize
        X = vectorizer.transform(df['combined_text'])

        # Predict
        probs = model.predict_proba(X)[:, 1]
        preds = model.predict(X)

        df['fraud_probability'] = probs
        df['prediction'] = preds

        return df

    except Exception as e:
        st.error(f"❌ Error: {str(e)}")
        return None

# File uploader
uploaded_file = st.file_uploader("📤 Upload a job listings CSV", type=["csv"])

if uploaded_file is not None:
    df = predict_jobs(uploaded_file)
    if df is not None:
        # Display top 20 rows
        display_cols = ['title', 'location', 'fraud_probability', 'prediction']
        display_cols = [col for col in display_cols if col in df.columns]
        st.subheader("📋 Preview Table (Top 20 rows)")
        st.dataframe(df[display_cols].head(20))

        # Pie chart
        st.subheader("🥧 Real vs Fake Job Listings")
        counts = df['prediction'].value_counts()
        labels = ['Real', 'Fake']
        sizes = [counts.get(0, 0), counts.get(1, 0)]
        colors = ['green', 'red']
        fig1, ax1 = plt.subplots()
        ax1.pie(sizes, labels=labels, autopct='%1.1f%%', colors=colors)
        ax1.axis('equal')
        st.pyplot(fig1)

        # Histogram using Matplotlib only
        st.subheader("📊 Fraud Probability Distribution")
        fig2, ax2 = plt.subplots()
        ax2.hist(df['fraud_probability'], bins=20, color='orange', alpha=0.7)
        ax2.set_title("Fraud Probability Distribution")
        ax2.set_xlabel("Fraud Probability")
        ax2.set_ylabel("Number of Listings")
        st.pyplot(fig2)
