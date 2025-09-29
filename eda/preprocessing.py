import pandas as pd
import re
import joblib
import string
import os
from sklearn.feature_extraction.text import TfidfVectorizer
# Load the CSV (update path if needed)
df = pd.read_csv('datasets/fake_job_postings.csv')

# Drop irrelevant or low-value columns
df.drop(['job_id', 'salary_range', 'telecommuting', 'has_company_logo', 'has_questions'], axis=1, inplace=True)

# Fill NaNs in text columns with empty string
text_columns = ['title', 'location', 'department', 'company_profile', 'description',
                'requirements', 'benefits', 'required_experience', 'required_education',
                'industry', 'function']

for col in text_columns:
    df[col] = df[col].fillna('')

# Clean text function
def clean_text(text):
    text = text.lower()
    text = re.sub(r'.*?', '', text)
    text = re.sub(r'https?://\S+|www\.\S+', '', text)
    text = re.sub(r'<.*?>+', '', text)
    text = re.sub(r'[%s]' % re.escape(string.punctuation), '', text)
    text = re.sub(r'\n', ' ', text)
    text = re.sub(r'\w*\d\w*', '', text)
    return text

# Combine important columns into one
df['combined_text'] = (
    df['title'] + ' ' + df['company_profile'] + ' ' +
    df['description'] + ' ' + df['requirements'] + ' ' +
    df['benefits'] + ' ' + df['industry'] + ' ' + df['function']
).apply(clean_text)

# Keep only necessary columns
df = df[['combined_text', 'fraudulent']]
def vectorizer():
        tfidf_vectorizer = TfidfVectorizer(
        max_features=5000,
        min_df=2,
        max_df=0.85,
        ngram_range=(1,2),
        norm='l2',
        sublinear_tf=True)
        tfidf_matrix = tfidf_vectorizer.fit_transform(df['combined_text'])
        tfidf_features = tfidf_vectorizer.get_feature_names_out()
        tfidf_df = pd.DataFrame(
            tfidf_matrix.toarray(),
            columns=tfidf_features
        )
        
        return tfidf_df,tfidf_vectorizer

# Ensure output directory exists
os.makedirs('datasets', exist_ok=True)

# Save cleaned data
df.to_csv('datasets/cleaned_fake_job_postings.csv', index=False)

print("✅ Preprocessing completed. Cleaned data saved to 'datasets/cleaned_fake_job_postings.csv'")