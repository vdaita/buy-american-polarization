import os
import json
import pandas as pd
import numpy as np
from tqdm import tqdm
from transformers import pipeline
import nltk
from nltk.corpus import stopwords

possible_classes = [
    'Fashion',
    'Electronics',
    'Home and Kitchen',
    'Health and Beauty',
    'Sports and Outdoors',
    'Books and Media',
    'Toys and Games',
    'Automotive',
    'Food and Beverage',
    'Miscellaneous'
]

nltk.download('stopwords')

stop_words = set(stopwords.words('english'))

def propagate_str_value_forward(df):
    df = df.copy()
    df.replace('', np.nan, inplace=True)  # Replace empty strings with NaN
    df = df.T.ffill(axis=0).T  # Forward fill NaN values
    df.fillna('', inplace=True)  # Fill remaining NaN with empty strings
    return df

def remove_stopwords(text):
    if isinstance(text, str):
        return ' '.join([word for word in text.split() if word.lower() not in stop_words])
    return text

website_df = pd.read_csv("website_data.csv")
website_df = propagate_str_value_forward(website_df)
selected_cols = reversed([col for col in website_df.columns if col.startswith("1") or col.startswith("2")])
website_df = website_df[selected_cols]

for col in tqdm(website_df.select_dtypes(include=['object']).columns):
    website_df[col] = website_df[col].apply(remove_stopwords)

classifier = pipeline("zero-shot-classification", model="distilbert-base-uncased")

website_classifications = []
if os.path.exists("website_classifications.json"):
    with open("website_classifications.json", "r") as f:
        website_classifications = json.load(f)

if not website_classifications:
    for i, row in tqdm(website_df.iterrows(), total=len(website_df)):
        text = row[-1]
        try:
            result = classifier(text, possible_classes)
            classification = result['labels'][0]
        except Exception as e:
            print(f"Error processing row {i}: {e}")
            classification = "Miscellaneous"  # Default class for errors
            
        website_classifications.append(classification)
        
    with open("website_classifications.json", "w") as f:
        json.dump(website_classifications, f)

website_df["category"] = website_classifications

website_df.to_csv("website_data_with_classifications.csv", index=False)