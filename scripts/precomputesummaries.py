import pandas as pd
from transformers import pipeline

df = pd.read_csv("newmetadata.csv")


model = pipeline(task="summarization",
                 model="facebook/bart-large-cnn", device="cuda:0")

summaries = []
for text in df['text']:
    summary = model([text])[0]['summary_text']
    summaries.append(summary)
    print(summary)


df2 = pd.DataFrame(summaries, columns=['summary'])
df2.to_csv("summaries")
