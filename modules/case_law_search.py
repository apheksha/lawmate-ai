from sentence_transformers import SentenceTransformer, util
import pandas as pd
import os

model = SentenceTransformer('all-MiniLM-L6-v2')

case_law_file = "data/case_law/case_law.csv"
if os.path.exists(case_law_file):
    df_cases = pd.read_csv(case_law_file)
    df_cases['embedding'] = df_cases['text'].apply(lambda x: model.encode(x, convert_to_tensor=True))
else:
    df_cases = pd.DataFrame(columns=['title','text','embedding'])

def search_case_law(query, top_k=3):
    query_emb = model.encode(query, convert_to_tensor=True)
    df_cases['score'] = df_cases['embedding'].apply(lambda x: util.cos_sim(query_emb, x).item())
    results = df_cases.sort_values('score', ascending=False).head(top_k)
    return results[['title','text','score']].to_dict(orient='records')
