# %%
import pandas as pd
import numpy as np
# Load model directly
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# %%
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
tokenizer = AutoTokenizer.from_pretrained("SravaniNirati/bert_fake_review_detection")
model = AutoModelForSequenceClassification.from_pretrained("SravaniNirati/bert_fake_review_detection").to(device)

# %%
inputs = tokenizer("This is super great product. Must Buy.", return_tensors="pt").to(device)
output = model(**inputs)

model.config.id2label

# %%
df = pd.read_csv("./data/fake_reviews_dataset.csv")
df.sample(7)

# %%
df.isna().sum()

# %%
df['label'] = df['label'].apply(lambda x: 1 if x == "CG" else 0)

# %%
df

# %%
# Predict using model in label_predicted

# Tokenize the review texts in batches for efficiency
batch_size = 64
preds = []

from tqdm import tqdm

for i in tqdm(range(0, len(df), batch_size), desc="Predicting batches"):
    batch_texts = df['text_'].iloc[i:i+batch_size].tolist()
    inputs = tokenizer(batch_texts, padding=True, truncation=True, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = model(**inputs)
        # Move tensor to CPU before converting to numpy
        batch_preds = torch.argmax(outputs.logits, dim=1).cpu().numpy()
        preds.extend(batch_preds)
df['label_predicted'] = [1 if x == 0 else 0 for x in preds]


# %%
df['label_predicted'] = df['label_predicted'].map({0: 1, 1: 0})

# %%
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

y_true = df['label']
y_pred = df['label_predicted']

accuracy = accuracy_score(y_true, y_pred)
precision = precision_score(y_true, y_pred)
recall = recall_score(y_true, y_pred)
f1 = f1_score(y_true, y_pred)

print(f"Accuracy:  {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall:    {recall:.4f}")
print(f"F1 Score:  {f1:.4f}")

# %%



