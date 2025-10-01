import gradio as gr
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Load model and vectorizer
model = joblib.load('fake_job_model.pkl')
vectorizer = joblib.load('tfidf_vectorizer.pkl')

def process_file(file):
    try:
        # Read CSV
        df = pd.read_csv(file.name)

        # Determine which text columns exist
        text_columns = [col for col in ['title', 'description', 'company_profile', 'requirements', 'benefits'] if col in df.columns]

        if not text_columns:
            return "No text columns found!", None, None

        # Combine text fields
        df['combined_text'] = df[text_columns].fillna('').agg(' '.join, axis=1)

        # Vectorize
        X = vectorizer.transform(df['combined_text'])

        # Predict
        probs = model.predict_proba(X)[:, 1]
        preds = model.predict(X)

        df['fraud_probability'] = probs
        df['prediction'] = preds

        # Prepare table preview
        display_cols = ['title', 'location', 'fraud_probability', 'prediction']
        display_cols = [col for col in display_cols if col in df.columns]
        preview_table = df[display_cols].head(20)  # show top 20 rows

        # Pie chart
        counts = df['prediction'].value_counts()
        labels = ['Real', 'Fake']
        sizes = [counts.get(0, 0), counts.get(1, 0)]
        colors = ['green', 'red']
        fig1, ax1 = plt.subplots()
        ax1.pie(sizes, labels=labels, autopct='%1.1f%%', colors=colors)
        ax1.axis('equal')
        pie_chart = fig1

        # Histogram
        fig2, ax2 = plt.subplots()
        sns.histplot(df['fraud_probability'], bins=20, kde=True, ax=ax2, color='orange')
        ax2.set_title("Fraud Probability Distribution")
        hist_chart = fig2

        return "✅ Processed Successfully!", preview_table, pie_chart, hist_chart

    except Exception as e:
        return f"❌ Error: {str(e)}", None, None, None

# Gradio Interface
with gr.Blocks() as demo:
    gr.Markdown("# 🛡️ Spot the Scam: Fake Job Detector")
    with gr.Row():
        file_input = gr.File(label="📤 Upload a job listings CSV", file_types=[".csv"])
    with gr.Row():
        status = gr.Label()
    with gr.Row():
        table = gr.Dataframe()
    with gr.Row():
        pie_output = gr.Plot()
        hist_output = gr.Plot()

    file_input.change(process_file, inputs=file_input, outputs=[status, table, pie_output, hist_output])

# Render requires this to use its assigned port
port = int(os.environ.get("PORT", 7860))
demo.launch(server_name="0.0.0.0", server_port=port)
