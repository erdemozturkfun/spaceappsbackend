import xml.etree.ElementTree as ET
import os
import pandas as pd
import re

from sentence_transformers import SentenceTransformer, util

model = SentenceTransformer("all-mpnet-base-v2")

phase_desc = {
    "Takeoff": "Rocket launches, ascent, liftoff",
    "Orbit": "Microgravity, orbital experiments, space station, cosmic radiation",
    "Moon Surface": "Moon surface operations, regolith, lunar habitat",
    "Mars Surface": "Mars surface, Mars Orbit, martian habitat, martian atmosphere"
}
phase_emb = {k: model.encode(v) for k, v in phase_desc.items()}


df = pd.read_csv('SB_publication_PMC.csv')
links = df['Link'].to_numpy()

ids = []

for link in links:
    match = re.search(r'PMC\d+', link)
    if match:
        id = match.group(0)
        ids.append(id)

links_dict = dict(zip(ids, links))

records = []

for filename in os.listdir("papersxml"):
    if not filename.endswith(".xml"):
        continue
    path = os.path.join("papersxml", filename)

    tree = ET.parse(path)
    root = tree.getroot()

    # Extract PMC ID from XML
    pmc_id = None
    article_id_tags = root.findall(".//article-id")
    for tag in article_id_tags:
        if tag.attrib.get("pub-id-type") == "pmcid":
            pmc_id = tag.text
            break
    if pmc_id is None:
        continue

    # Extract title
    title_tag = root.find(".//article-title")
    title = ET.tostring(title_tag, method="text",
                        encoding="unicode") if title_tag is not None else ""
    paper_emb = model.encode(title)

    date = root.find(".//pub-date")
    year = date.find("year")

    similarities = {phase: util.cos_sim(
        paper_emb, emb).item() for phase, emb in phase_emb.items()}
    assigned_phase = max(similarities, key=similarities.get)

    # Get URL from links_dict
    url = links_dict.get(pmc_id, "")

    records.append({"paper_id": pmc_id, "title": title,
                   "url": url, "assigned_phase": assigned_phase,  "year": year.text})
metadata_df = pd.DataFrame(records)

old_df = pd.read_csv("papers_chunks.csv")

new_df = old_df.merge(
    metadata_df[['paper_id', 'title', 'url', 'assigned_phase', 'year']], on='paper_id', how="left")

new_df.to_csv('newmetadata.csv')
