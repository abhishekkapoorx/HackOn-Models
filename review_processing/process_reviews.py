# %%
import pandas as pd
import numpy as np

# %%
rev1 = pd.read_csv("../reviews/1429_1.csv")
rev2 = pd.read_csv("../reviews/Datafiniti_Amazon_Consumer_Reviews_of_Amazon_Products_May19.csv")
rev3 = pd.read_csv("../reviews/Datafiniti_Amazon_Consumer_Reviews_of_Amazon_Products.csv")

# %%
# print(rev1.columns, rev2.columns, rev3.columns)
# Find common columns across all three dataframes using set intersection
common_columns = list(set(rev1.columns) & set(rev2.columns) & set(rev3.columns))
print("Common columns across all three dataframes:")
print(sorted(common_columns))
print(f"\nNumber of common columns: {len(common_columns)}")


# %%
# Combine all three dataframes using only the common columns
combined_df = pd.concat([rev1[common_columns], rev2[common_columns], rev3[common_columns]], 
                       ignore_index=True)

print(f"Combined dataframe shape: {combined_df.shape}")
print(f"Combined dataframe columns: {list(combined_df.columns)}")
combined_df.head()


# %%
combined_df.columns.sort_values()

# %%
review_df_sementic = combined_df[["reviews.title", "reviews.text", "reviews.rating"]]
review_df_sementic.sample(5)

# %%
review_df_sementic.isnull().sum()
review_df_sementic.dropna(inplace=True)
review_df_sementic.duplicated().sum()
review_df_sementic.drop_duplicates(inplace=True)

# %%
review_df_sementic.loc[:, "rating_semantic"] = review_df_sementic["reviews.rating"].apply(lambda x: 1 if x > 3 else -1 if x < 3 else 0)


# %%
import torch
# Use the yangheng/deberta-v3-large-absa-v1.1 model with proper tokenizer configuration
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline

# Load the model with trust_remote_code=True to handle custom tokenizer
model_name = "yangheng/deberta-v3-base-absa-v1.1"
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
model = AutoModelForSequenceClassification.from_pretrained(model_name, trust_remote_code=True)

# Create aspect-based sentiment pipeline
absa_pipe = pipeline(
    "text-classification",
    model=model,
    tokenizer=tokenizer,
    return_all_scores=True,
    trust_remote_code=True
)

# Function to get sentiment scores using the ABSA model
def get_absa_sentiment(text):
    try:
        result = absa_pipe(text[:512])  # Truncate to avoid token limits
        # For ABSA model, we need to check the label mapping
        # Usually positive sentiment has higher score
        return result[0]
    except Exception as e:
        print(f"Error processing text: {e}")
        return [{"label": "neutral", "score": 0.5}]

# Function to convert ABSA result to binary rating
def absa_to_rating(absa_result, threshold=0.5):
    # Find the highest scoring label
    best_score = max(absa_result, key=lambda x: x['score'])
    return 1 if best_score['score'] > threshold else 0


# %%
%%time
text = review_df_sementic['reviews.text'].sample().to_list()[0]
print(text, absa_pipe(text), get_absa_sentiment(text))


# %% [markdown]
# 

# %%
# Function to process a list of texts and return ratings
def get_rating_semantic_batch(text_list):
    ratings = []
    for text in text_list:
        try:
            absa_result = get_absa_sentiment(text)
            # Convert to -1, 0, 1 rating based on sentiment scores
            # Find the highest scoring label by comparing scores directly
            max_score = 0
            best_label = 'neutral'
            for item in absa_result:
                if abs(item['score']) > abs(max_score):
                    max_score = item['score']
                    best_label = item['label']
            
            
            if best_label == 'Positive':
                ratings.append(1)
            elif best_label == 'Negative':
                ratings.append(-1)
            else:
                ratings.append(0)
        except Exception as e:
            print(f"Error processing text: {e}")
            ratings.append(0)
    return ratings


# %%
sample = review_df_sementic.sample(1000)

# %%


# %%
sample.loc[:, "rating_semantic_generated"] = get_rating_semantic_batch(sample["reviews.text"].tolist())

# %%
sample.loc[sample['rating_semantic_generated'] == 0]

# %%
accuracy = (sample["rating_semantic_generated"] == sample["rating_semantic"]).mean()
accuracy

# %%
combined_df.isnull().sum()
combined_df.drop(columns=["reviews.numHelpful", "reviews.id", "name", "reviews.doRecommend"], inplace=True)
combined_df.dropna()
combined_df.shape

# %%
combined_df.duplicated().sum()
combined_df.drop_duplicates(inplace=True)

# %%
combined_df.to_csv("../reviews/combined_reviews.csv")

# %% [markdown]
# # Aspect generation for each product

# %%
top_5_reviewed = combined_df["id"].value_counts()[:5].index
top_5_reviewed

# %%
import pandas as pd
from pyabsa import AspectTermExtraction as ATEPC
from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from langchain.chains import LLMChain
import os

# STEP 1: Get product ID from user
product_id = "AVphgVaX1cnluZ0-DR74"

# STEP 2: Load reviews using pandas
df = pd.read_csv("..\\reviews\\combined_reviews.csv")  # replace with your CSV path
product_reviews = df[df["id"] == product_id]["reviews.text"].dropna().tolist()

# STEP 3: Load ATEPC aspect-based sentiment model
aspect_extractor = ATEPC.AspectExtractor(
    checkpoint="english",  # You can also try "english"
    auto_device=True  # uses GPU if available
)

# STEP 4: Extract aspects and sentiment
extracted = aspect_extractor.extract_aspect(
    inference_source=product_reviews,  # Add the missing inference_source parameter
    examples=product_reviews,
    pred_sentiment=True,
    print_result=False
)

# STEP 5: Aggregate aspects by sentiment
from collections import defaultdict
aspect_summary = defaultdict(list)

for entry in extracted:
    for asp, sent in zip(entry['aspect'], entry['sentiment']):
        aspect_summary[sent].append(asp)

# Keep only top 10 unique aspects per sentiment
def top_aspects(aspect_list):
    from collections import Counter
    return [item for item, _ in Counter(aspect_list).most_common(10)]

positive_aspects = top_aspects(aspect_summary["Positive"])
negative_aspects = top_aspects(aspect_summary["Negative"])

# STEP 6: Use LangChain + Groq to give suggestions
os.environ["GROQ_API_KEY"] = "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"  # Replace with your Groq key

llm = ChatGroq(model="llama3-70b-8192")

prompt = PromptTemplate(
    input_variables=["positives", "negatives"],
    template="""
You are a product advisor. Based on these positive and negative aspects from product reviews:

Positive aspects: {positives}
Negative aspects: {negatives}

Give sellers two lists:
1. Improvements they should make (based on negative aspects).
2. Strengths they should highlight (based on positive aspects).

Return the output as a Python dictionary with keys 'positive' and 'negative', each having list of strings as suggestions.
"""
)

chain =  prompt | llm
response = chain.invoke({
    "positives": ", ".join(positive_aspects),
    "negatives": ", ".join(negative_aspects)
})

# Final Output
print("\n🧠 Suggestion Summary:\n")
print(response)



