
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
import os

os.makedirs("../outputs", exist_ok=True)


df = pd.read_csv("datasets/fake_job_postings.csv")

#  Basic info
print(df.shape)
print(df.columns)
print(df.isnull().sum())

#  Drop unnecessary columns
cols_to_drop = ['job_id', 'salary_range', 'telecommuting', 'has_company_logo', 'has_questions']
df.drop(columns=cols_to_drop, inplace=True, errors='ignore')

#  Fill missing values
df.fillna('', inplace=True)

#  Combine text columns
df['combined_text'] = df[['title', 'company_profile', 'description', 'requirements', 'benefits']].agg(' '.join, axis=1)

#  Class distribution plot
plt.figure(figsize=(6,4))
sns.countplot(data=df, x='fraudulent', palette='Set2')
plt.title('Class Distribution (0 = Real, 1 = Fraudulent)')
plt.xlabel('Fraudulent')
plt.ylabel('Count')
plt.tight_layout()
plt.savefig("../outputs/class_distribution.png")
plt.show()

#  WordCloud function
def plot_wordcloud(label, title, filename):
    text = " ".join(df[df['fraudulent'] == label]['combined_text'])
    wc = WordCloud(width=800, height=400, background_color='white', stopwords='english').generate(text)
    plt.figure(figsize=(10,5))
    plt.imshow(wc, interpolation='bilinear')
    plt.axis('off')
    plt.title(title)
    plt.tight_layout()
    plt.savefig(f"../outputs/{filename}")
    plt.show()

plot_wordcloud(0, "WordCloud: Real Job Listings", "wordcloud_real.png")
plot_wordcloud(1, "WordCloud: Fake Job Listings", "wordcloud_fake.png")

#  Top 10 fake titles
plt.figure(figsize=(10,4))
df[df['fraudulent']==1]['title'].value_counts().head(10).plot(kind='bar', color='tomato')
plt.title('Top 10 Titles in Fake Jobs')
plt.ylabel('Count')
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig("../outputs/top_fake_titles.png")
plt.show()

#  Top 10 fake locations
plt.figure(figsize=(10,4))
df[df['fraudulent']==1]['location'].value_counts().head(10).plot(kind='bar', color='purple')
plt.title('Top Locations of Fake Jobs')
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig("../outputs/top_fake_locations.png")
plt.show()

#  Save cleaned data (optional)
df.to_csv("../datasets/cleaned_fake_job_postings.csv", index=False)
