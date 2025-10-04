from Bio import Entrez
import numpy as np
import pandas as pd
import re
import time
import xml.etree.ElementTree as ET

df = pd.read_csv('SB_publication_PMC.csv')
links = df['Link'].to_numpy()

ids = []

for link in links:
    match = re.search(r'PMC\d+', link)
    if match:
        id = match.group(0)
        ids.append(id)


ids = [ids[i:i+4]for i in range(0, len(ids), 4)]

Entrez.email = "erdemozturk200@gmail.com"

for batch in ids:
    ids_str = ",".join(batch)
    try:
        print(ids_str)
        handle = Entrez.efetch(db='pmc', id=ids_str,
                               rettype="xml", retmode="text")
        xml = handle.read()

        handle.close()

        root = ET.fromstring(xml)

        for article in root.findall(".//article"):
            pmc_id = None
            for id in article.findall(".//article-id"):
                if id.get("pub-id-type") == "pmcid":

                    pmc_id = id.text
                    break
            if pmc_id == None:
                print("No pmc_id")
                break

            filename = f"{pmc_id}.xml"
            with open(filename, "w", encoding="utf-8") as f:
                f.write(ET.tostring(article, encoding="unicode"))
            print(f"Saved {filename}")
            break
    except Exception as e:
        attempt += 1

        print(e)
        if attempt > 3:
            raise
        backoff = 2**attempt
        time.sleep(backoff)
