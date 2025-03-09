import os
import logging
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
import pickle

logger = logging.getLogger(__name__)


# Load a text embedding model
embed_model = SentenceTransformer("all-MiniLM-L6-v2")

def index_notes(vault_path: str, dest_path: str = "./index/"):
    logger.info("Indexing notes...")

    # Store notes and filenames
    note_texts = []
    filenames = []

    # Load all Markdown files
    for root, _, files in os.walk(vault_path):
        for file in files:
            if file.endswith(".md"):
                file_path = os.path.join(root, file)
                with open(file_path, "r", encoding="utf-8") as f:
                    text = f.read()
                    note_texts.append(text)
                    filenames.append(file_path)

    logger.info("Total number of notes: %d", len(note_texts))

    # Convert notes to embeddings
    logger.info("Creating embeddings...")
    embeddings = embed_model.encode(note_texts)

    # Store in FAISS (vector database)
    logger.info("Creating FAISS index...")
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(np.array(embeddings))

    # Save the FAISS index and filenames
    faiss.write_index(index, os.path.join(dest_path, "obsidian_notes.index"))
    with open(os.path.join(dest_path, "filenames.pkl"), "wb") as f:
        pickle.dump(filenames, f)

    print("✅ Notes indexed successfully!")

def main():
    VAULT_PATH = "/Users/seletz/develop/notes"
    index_notes(VAULT_PATH)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(filename)s:%(lineno)d - %(message)s")
    main()
