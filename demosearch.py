import networkx as nx
from transformers import pipeline, Pipeline


def search(embed, index, metadata_df, top_k=5, mission_phases=["Orbit", "Takeoff", "Moon Surface", "Mars Surface"], years="1945#2025"):
    years = years.split("-")
    startingYear = int(years[0])
    endingYear = int(years[1])

    D, I = index.search(embed, top_k+1)
    results = []
    for idx in I[0]:
        row = metadata_df.iloc[idx]

        if mission_phases and row.get('assigned_phase') not in mission_phases:
            continue
        if row.get('year') not in range(startingYear, endingYear, 1):
            continue

        results.append({"paper_id": row['paper_id'], "section": row
                       ['section'], "text": row['text'], "url": row["url"], "title": row["title"], "assigned_phase": row.get('assigned_phase'), "year": row.get('year'), "embedid": idx})

    return results


def deduplicate_results_by_paper(search_results):

    seen = set()
    deduped = []

    for res in search_results:
        pid = res['paper_id']
        if pid not in seen:
            deduped.append(res)
            seen.add(pid)

    return deduped


def build_subgraph_from_search(search_results, edges_dict, metadata_df, max_neighbors=5):

    G = nx.DiGraph()
    paper_to_idx = {}
    for res in search_results:
        paper_id = res['paper_id']
        if paper_id not in paper_to_idx:
            node_id = len(paper_to_idx)
            paper_to_idx[paper_id] = node_id
            G.add_node(node_id, paper_id=paper_id,
                       title=res["title"], url=res["url"], sections=[res.get('section')] if 'section' in res else [])
        else:
            node_id = paper_to_idx[paper_id]
            if 'section' in res and res['section'] not in G.nodes[node_id]['section']:
                G.nodes[node_id]['sections'].append(res['section'])
        idx = res['embedid']
        neighbors = edges_dict.get(str(idx), [])[:max_neighbors]
        neighbor_papers = set()
        for n_idx in neighbors:
            n_idx = int(n_idx)

            neighbor_meta = metadata_df.iloc[n_idx]
            n_paper_id = neighbor_meta['paper_id']

            if n_paper_id not in paper_to_idx:
                n_node_id = len(paper_to_idx)
                paper_to_idx[n_paper_id] = n_node_id
                G.add_node(n_node_id, paper_id=n_paper_id,
                           title=neighbor_meta.get('title', ''),
                           url=neighbor_meta.get('url', ''))
            else:
                n_node_id = paper_to_idx[n_paper_id]
            G.add_edge(node_id, n_node_id)

    from networkx.readwrite import json_graph
    graph_json = json_graph.node_link_data(G)
    return graph_json


def createEdges(embed, index, k):
    edges = {}
    for idx in range(len(embed)):
        D, I = index.search(embed[idx].reshape(1, -1), k+1)
        edges[idx] = I[0][1:].tolist()
    return edges


def createSummary(results, summary_df):
    summaries = []
    for result in results:

        summary = summary_df.iloc[int(result['embedid'])]['summary']
        summaries.append(
            f"This paper ({result['paper_id']}, {result['section']}) says {summary}.")

    return summaries


def get_full_paper(metadata_df, paper_id):
    sections = metadata_df[metadata_df['paper_id'] == paper_id]['embedid']
    sections
    return sections


def summarize_full_paper(sections, summary_df, summarizer):

    summaries = []
    for chunk in sections:
        summary = summary_df.iloc[int(chunk)]['summary']
        summaries.append(summary)

    summaries = " ".join(summaries)
    full_summary = summarizer(summaries)[0]['summary_text']
    return full_summary
