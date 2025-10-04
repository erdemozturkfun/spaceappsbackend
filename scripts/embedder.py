import pandas as pd
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from transformers import BartForConditionalGeneration, BartTokenizer

# ----------------------------
# 1. Load CSV with chunks
# ----------------------------
df = pd.read_csv("papers_chunks.csv")
texts = df['text'].tolist()

# ----------------------------
# 2. Generate embeddings for chunks (SBERT)
# ----------------------------
embed_model_name = "sentence-transformers/all-MiniLM-L6-v2"
embed_model = SentenceTransformer(embed_model_name)
embeddings = embed_model.encode(
    texts, convert_to_numpy=True, show_progress_bar=True)


np.save("embeddings.npy", embeddings)
