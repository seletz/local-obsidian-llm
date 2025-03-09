import logging
import pickle

import faiss
import numpy as np
from flask import Flask, request, jsonify
from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)

# Load FAISS index and filenames
index = faiss.read_index("index/obsidian_notes.index")
with open("index/filenames.pkl", "rb") as f:
    filenames = pickle.load(f)

# Load embedding model
embed_model = SentenceTransformer("all-MiniLM-L6-v2")

# LM Studio API URL
LM_STUDIO_API_URL = "http://localhost:8080/v1/completions"

app = Flask(__name__)


# Function to retrieve relevant notes
def search_notes(query, top_k=3):
    query_embedding = embed_model.encode([query])
    _, indices = index.search(np.array(query_embedding), top_k)
    return [filenames[i] for i in indices[0]]


@app.route("/function_call", methods=["POST"])
def handle_function_call():
    data = request.json
    func_name = data.get("function")
    logger.info(f"Handling function call: {func_name}")

    if func_name == "search_notes":
        query = data["parameters"]["query"]
        relevant_notes = search_notes(query)

        # Load the text of the retrieved notes
        note_contents = "\n\n".join([open(f, "r").read() for f in relevant_notes])

        return jsonify({
            "status": "success",
            "notes": note_contents
        })

    logger.error(f"Unknown function: {func_name}")
    return jsonify({"status": "error", "message": "Unknown function"}), 400


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(filename)s:%(lineno)d - %(message)s")
    app.run(host="0.0.0.0", port=5002)