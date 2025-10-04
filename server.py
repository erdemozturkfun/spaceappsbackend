from fastapi import FastAPI
import demosearch
from faiss import IndexFlatL2
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline
import numpy as np
import pandas as pd
import os
import json
app = FastAPI()

embeddings = None
index = None
metadata_df = None
summarizer = None
edges = None


@app.on_event("startup")
async def startup_event():
    global embeddings, index, metadata_df, summarizer, edges
    import numpy as np
    import pandas as pd
    from faiss import IndexFlatL2
    from transformers import pipeline
    import demosearch

    embeddings = np.load('embeddings.npy')
    metadata_df = pd.read_csv("newmetadata.csv")

    index = IndexFlatL2(embeddings.shape[1])
    index.add(embeddings)

    if not os.path.exists("edges.json"):
        edges = demosearch.createEdges(embeddings, index, 5)
        with open("edges.json", "w") as f:
            json.dump(edges, f)
    else:
        with open("edges.json", "r") as f:
            edges = json.load(f)

    summarizer = pipeline(task="summarization",
                          model="facebook/bart-large-cnn")
    print("✅ Startup finished, model loaded")


@app.get("/")
def read_root(args: str = "Hello"):
    return args


@app.get("/summary")
def get_Summary(query: str):

    print("hi")
    searches = demosearch.search(embeddings, index, metadata_df)
    return demosearch.createSummary(searches, summarizer)


@app.get("/graph")
def createGraph(query: str):
    searches = demosearch.search(embeddings, index, metadata_df)
    searches = demosearch.deduplicate_results_by_paper(searches)
    return demosearch.build_subgraph_from_search(searches, edges, metadata_df)
