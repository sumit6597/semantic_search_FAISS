# Semantic Search with Flask and Sentence Transformers

This repository contains a simple web application for semantic search using Flask, Sentence Transformers, and Faiss. The application allows users to input a query, and it retrieves the most relevant documents from a preprocessed dataset.


## Installation

1. **Clone the repository:**
    ```bash
    git clone <SSH/HTTPS>
    cd <dir>
    ```

2. **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

3. **Download the Sentence Transformers model and place it in the 'cache' folder.**

4. **Provide your dataset (e.g., Gesund.csv) and update the 'path' variable in the script accordingly.**

## Usage

1. **Run the Flask application:**
    ```bash
    python run.py
    ```

2. **Open your web browser and navigate to [http://127.0.0.1:5000/](http://127.0.0.1:5000/).**

3. **Enter a query in the search bar and submit the form.**

4. **View the search results, including the most relevant documents based on semantic similarity.**

## Project Structure

- **run.py:** Flask application script containing routes for home page and search functionality.
- **templates:** HTML templates for rendering the web pages.
- **cache:** Folder to store cached models and data.
- **dense_vectors.pkl:** Cached dense vectors of document embeddings.
- **faiss_index.index:** Cached Faiss index for fast similarity search.
- **Gesund.csv:** Example dataset (replace with your own dataset).
- **requirements.txt:** List of Python dependencies.

## Implementation Details

- The application uses the Sentence Transformers library to encode text into dense vectors.
- Faiss is employed to create an index for efficient similarity search.
- The provided dataset (Gesund.csv) is preprocessed, and the script caches document embeddings and the Faiss index for faster retrieval.
- Users can input queries through the web interface, and the application returns the most relevant documents based on semantic similarity.

Feel free to customize the code according to your specific use case and dataset.
