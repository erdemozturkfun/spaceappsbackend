from fastapi import FastAPI
import demosearch
from faiss import IndexFlatL2
from transformers import pipeline
from sentence_transformers import SentenceTransformer
import numpy as np
import pandas as pd
import os
import json

app = FastAPI()


@app.on_event("startup")
async def startup_event():
    embeddings = np.load('embeddings.npy')
    metadata_df = pd.read_csv("newmetadata.csv")
    summaries = pd.read_csv("summaries.csv")

    embedder = SentenceTransformer("all-MiniLM-L6-v2")

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

    app.state.embeddings = embeddings
    app.state.index = index
    app.state.metadata_df = metadata_df
    app.state.summarizer = summarizer
    app.state.edges = edges
    app.state.embedder = embedder
    app.state.summaries = summaries
    print("✅ Startup finished, model loaded")


@app.get("/")
def read_root(args: str = "Hello"):
    return args


@app.get("/summary")
def get_summary(query: str, year: str, filter: str):

    mission_phases = filter.split(',')

    embedder = app.state.embedder
    index = app.state.index
    metadata_df = app.state.metadata_df
    summaries = app.state.summaries

    query_embed = embedder.encode([query])

    searches = demosearch.search(
        query_embed, index, metadata_df, top_k=15, years=year, mission_phases=mission_phases)
    searches = demosearch.deduplicate_results_by_paper(searches)

    return demosearch.createSummary(searches, summaries)


@app.get("/graph")
def createGraph(query: str, year: str, filter: str):

    mission_phases = filter.split(',')
    embedder = app.state.embedder
    index = app.state.index
    metadata_df = app.state.metadata_df
    edges = app.state.edges
    query_embed = embedder.encode([query])
    searches = demosearch.search(
        query_embed, index, metadata_df, years=year, mission_phases=mission_phases, top_k=15)
    deduped = demosearch.deduplicate_results_by_paper(searches)
    return demosearch.build_subgraph_from_search(deduped, edges, metadata_df)


@app.get("/fullsummary")
def summarisePaper(id: str):
    summaries = app.state.summaries
    summarizer = app.state.summarizer

    metadata_df = app.state.metadata_df
    paper = demosearch.get_full_paper(metadata_df, id)
    return demosearch.summarize_full_paper(paper, summary_df=summaries, summarizer=summarizer)
