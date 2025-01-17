import streamlit as st
from joblib import load
from sklearn.metrics.pairwise import cosine_similarity
from transformers import DistilBertTokenizer, DistilBertModel
import torch
import numpy as np
import re

df = load("dataframe.joblib")

st.title("E-commerce Product Recommendation")

device = "cuda" if torch.cuda.is_available() else "cpu"



def cleaned_text(text):
    text_cleaned = re.sub(r'[^a-zA-Z0-9\s]', '', text).lower()
    return text_cleaned


model = DistilBertModel.from_pretrained('distilbert-base-uncased').to(device)
tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
def get_text_embedding(text):
    tokenized_text = tokenizer.encode(text, max_length=216, truncation=True, return_tensors='pt').to(device)
    with torch.no_grad():
        model_output = model(tokenized_text)
    cls_embedding = model_output.last_hidden_state[:,0,:]
    return cls_embedding.cpu().numpy()

def cosine_similarities(embedding, embeddings):
    similarities = cosine_similarity(embedding.reshape(1, -1), np.vstack(embeddings)).flatten()
    return similarities


def recommendation_function(text, df, top_n):
    text_cleaned = cleaned_text(text)
    embedding = get_text_embedding(text_cleaned)
    embeddings = df['text_embedding'].tolist()
    similarities = cosine_similarities(embedding, embeddings)
    df['similarity'] = similarities
    df_sorted = df.sort_values(by='similarity', ascending=False)
    recommendation = df_sorted.head(top_n)
    return recommendation[['title', 'imgUrl', 'productURL', 'stars', 'reviews', 'price',
                           'listPrice', 'category_id', 'isBestSeller', 'boughtInLastMonth',
                           'category_name', 'similarity']]


def display_recommendations_streamlit_alternative(recommendations):
    # Limit to 10 products only
    recommendations = recommendations.iloc[:40]

    # Display in rows of 5 products per row
    for row in range(0, len(recommendations), 4):
        cols = st.columns(4)  # Create 5 columns in the row
        for col, i in zip(cols, range(row, row + 4)):
            if i < len(recommendations):
                recommendation = recommendations.iloc[i]

                # Retrieve product details
                title = recommendation['title']
                image = recommendation['imgUrl']
                stars = recommendation['stars']
                reviews = recommendation['reviews']
                price = recommendation['price']
                boughtInLastMonth = recommendation['boughtInLastMonth']
                category_name = recommendation['category_name']
                product_url = recommendation['productURL']

                # Display product in a card-like layout
                with col:
                    st.markdown(
                        f"""
                        <div style="border: 1px solid #ddd; border-radius: 8px; padding: 10px; text-align: center; background-color: #f9f9f9;">
                            <img src="{image}" alt="{title}" style="width: 100px; height: 100px; object-fit: cover; margin-bottom: 10px; border-radius: 4px;">
                            <h4 style="font-size: 14px; margin-bottom: 5px; color: black;">{title}</h4>
                            <p style="font-size: 12px; margin: 2px 0; color: black;"><strong>Price:</strong> {price}</p>
                            <a href="{product_url}" target="_blank" style="font-size: 12px; color: #007bff; text-decoration: none;">View Product</a>
                        </div>
                        """,
                        unsafe_allow_html=True
                    )


# Input field with a placeholder
user_input = st.text_input("Search for a product:", placeholder="Type a product name here")

# Slider to select the number of recommendations (default is 10)
top_n = 8

# Check if user has provided input
if user_input.strip():  # Proceed only if input is not empty or whitespace
    with st.spinner("Finding recommendations..."):
        recommendations = recommendation_function(user_input, df, top_n)  # Get recommendations
        if not recommendations.empty:
            display_recommendations_streamlit_alternative(recommendations)  # Display the products
        else:
            st.warning("No recommendations found for your query. Please try a different search.")
else:
    # If the input is empty, display a message
    st.warning("Please enter a product name to get recommendations.")