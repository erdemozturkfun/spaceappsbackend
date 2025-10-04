
import os
import pandas as pd
import xml.etree.ElementTree as ET
from transformers import AutoTokenizer

# ----------------------------
# 1. Config
# ----------------------------
XML_FOLDER = "papersxml"    # folder with PMC XML files
CHUNK_SIZE = 512             # max tokens per chunk
CHUNK_OVERLAP = 50           # tokens to overlap between chunks
OUTPUT_CSV = "papers_chunks.csv"

embed_model_name = "sentence-transformers/all-MiniLM-L6-v2"
tokenizer = AutoTokenizer.from_pretrained(embed_model_name)


def extract_sections(xml_file):
    """Returns dict: {section_name: text}"""
    tree = ET.parse(xml_file)
    root = tree.getroot()
    # adjust namespace if needed
    ns = {'ns': 'NASABIO'}

    sections = {}

    # Common sections
    for sec_name in ['abstract', 'introduction', 'methods', 'results', 'discussion', 'conclusion']:
        texts = []
        # Try abstract separately as it may be <abstract> outside <body>
        if sec_name == 'abstract':
            for abstract in root.findall(".//abstract"):
                texts.append(' '.join(abstract.itertext()))
        else:
            for sec in root.findall(f".//sec[title='{sec_name.capitalize()}']"):
                texts.append(' '.join(sec.itertext()))
        if texts:
            sections[sec_name] = ' '.join(texts)
    return sections


def chunk_text(text, chunk_size=512, overlap=50):
    """Returns list of text chunks (token-limited)"""
    chunks = []
    start = 0
    # Tokenize once as IDs
    token_ids = tokenizer(text, add_special_tokens=False)['input_ids']

    while start < len(token_ids):
        end = start + chunk_size
        chunk_ids = token_ids[start:end]
        chunk_text = tokenizer.decode(chunk_ids, skip_special_tokens=True)
        chunks.append(chunk_text)
        start += chunk_size - overlap
    return chunks


rows = []
for xml_file in os.listdir(XML_FOLDER):
    if not xml_file.endswith(".xml"):
        continue
    file_path = os.path.join(XML_FOLDER, xml_file)
    paper_id = xml_file.replace(".xml", "")
    sections = extract_sections(file_path)

    for section_name, section_text in sections.items():
        if not section_text.strip():
            continue
        text_chunks = chunk_text(section_text, CHUNK_SIZE, CHUNK_OVERLAP)
        for i, chunk in enumerate(text_chunks):
            rows.append({
                "paper_id": paper_id,
                "section": section_name,
                "chunk_id": i,
                "text": chunk
            })


df = pd.DataFrame(rows)
df.to_csv(OUTPUT_CSV, index=False)
print(f"Saved {len(df)} chunks to {OUTPUT_CSV}")
