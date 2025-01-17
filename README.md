# **Ecommerce Product Recommendation System**
Welcome to the **Ecommerce Product Recommendation System**, a project leveraging Large languages Model and Natural language processing (NLP) techniques to deliver personalized product recommendations. 🚀
![Uploading reccomend ecommerce products.gif…](https://github.com/RickyDoan/MMLs-NLP-Recommend-E-commerce-Products/blob/main/reccomend%20ecommerce%20products.gif)

## **About the Project**
This project focuses on building a recommendation system for Ecommerce product data, with features such as:
- **Data Cleaning and Preprocessing**: Cleaning text data for better analysis, handling missing values, and removing duplicates.
- **Product Embeddings**: Using `DistilBERT` from the Hugging Face `transformers` library to generate high-quality embeddings for product titles. These embeddings are used to capture semantic similarity between products.
- **Recommendation Engine**: A function that utilizes cosine similarity to identify the top recommendations for a given product or search text.
- **Intuitive Visual Display**: Recommendations are displayed with thumbnail images, metadata, and similarity scores for easy browsing.

## **Key Features**
1. **Data Cleaning Pipeline**:
    - Handles missing and duplicate values.
    - Supports text cleaning with regular expressions.
    - Prepares well-structured data ready for recommendation.

2. **Text Embeddings**:
    - Utilizes `DistilBERT` for text embedding generation.
    - Extracts semantic meaning for more accurate product similarity measurements.

3. **Recommendation Functionality**:
    - Input any product description, and the system retrieves the most relevant products based on semantic similarity.
    - Highly versatile, works with any dataset following the same format.

4. **Interactive Display**:
    - A user-friendly visual representation of recommendations including product images, prices, reviews, and similarity scores.
    - Designed for seamless exploration.

## **Tech Stack Used**
- **Frameworks**:
    - `Streamlit` (optional) for deploying and visualizing the recommendations interactively.

- **Notebook Workflows**:
    - `Jupyter Notebook` for implementing, testing, and visualizing computations.

## **How It Works**
1. **Input Data**: The system cleans input product data
2. **Generate Embeddings**: Each product description is transformed into a numerical embedding via `DistilBERT`.
3. **Compute Similarity**: For a given product title or description, the system calculates cosine similarity across all embeddings.
4. **Output Recommendations**: The top products with the highest similarity scores are presented in a visually intuitive format.

## **Use Case Example**
Input: Searching for products related to **"Kids' educational toys for learning colors and shapes"**.
Output: A list of **top 10 recommendations** with metadata such as:
- Product name
- Rating and reviews
- Price and category
- Thumbnails for better visualization
