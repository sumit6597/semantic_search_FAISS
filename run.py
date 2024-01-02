from flask import Flask, render_template, request
import pandas as pd
from sentence_transformers import SentenceTransformer
import numpy as np
import faiss
from sentence_transformers import InputExample
import os
import pickle

app = Flask(__name__, template_folder='/Users/sumitkumar/Desktop/semantic-search/templates')

cache_dir = "./cache"
if not os.path.exists(cache_dir):
    os.makedirs(cache_dir)

path = "/Users/sumitkumar/Desktop/semantic-search/Gesund.csv"
pdf = pd.read_csv(path)
pdf["id"] = pdf.index
pdf_subset = pdf.head(1000)

def example_create_fn(doc1: pd.Series) -> InputExample:
    if pd.isnull(doc1["text_embedded"]):
        return InputExample(texts=[doc1["text_ocr"]])
    else:
        return InputExample(texts=[doc1["text_embedded"]])
    


faiss_train_examples = pdf_subset.apply(lambda x: example_create_fn(x), axis=1).tolist()

model = SentenceTransformer("all-MiniLM-L6-v2", cache_folder="./cache")

def encode_title_or_text(row):
    if pd.isnull(row['text_embedded']):
        return model.encode(row['text_ocr'])
    else:
        return model.encode(row['text_embedded'])

dense_vectors_file = 'dense_vectors.pkl'
faiss_index_file = 'faiss_index.index'

# Load dense vectors from a file if it exists, otherwise compute and save them
if os.path.exists(dense_vectors_file):
    with open(dense_vectors_file, 'rb') as file:
        faiss_title_embedding = pickle.load(file)
else:
    faiss_title_embedding = np.vstack(pdf_subset.apply(encode_title_or_text, axis=1).values)
    faiss.normalize_L2(faiss_title_embedding)
    with open(dense_vectors_file, 'wb') as file:
        pickle.dump(faiss_title_embedding, file)

# Load Faiss index from a file if it exists, otherwise compute and save it
if os.path.exists(faiss_index_file):
    index_content = faiss.read_index(faiss_index_file)
else:
    content_encoded_normalized = faiss_title_embedding.copy()
    faiss.normalize_L2(content_encoded_normalized)
    index_content = faiss.IndexIDMap(faiss.IndexFlatIP(len(faiss_title_embedding[0])))
    index_content.add_with_ids(content_encoded_normalized, np.array(pdf_subset.id.values).astype("int"))
    faiss.write_index(index_content, faiss_index_file)

pdf_to_index = pdf_subset.set_index(["id"], drop=False)

def search_content(query, pdf_to_index, k=3):
    query_vector = model.encode([query])
    faiss.normalize_L2(query_vector)

    top_k = index_content.search(query_vector, k)
    ids = top_k[1][0].tolist()
    similarities = top_k[0][0].tolist()
    results = pdf_to_index.loc[ids]
    results["similarities"] = similarities
    return results

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/search', methods=['POST'])
def search():
    query = request.form['query']
    results_df = search_content(query, pdf_to_index, k=5)
    return render_template('result.html', tables=[results_df.to_html(classes='data')], titles=results_df.columns.values)

if __name__ == '__main__':
    app.run(debug=True)
