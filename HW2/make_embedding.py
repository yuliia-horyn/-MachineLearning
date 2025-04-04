import time
import pandas as pd
import streamlit as st
from tqdm import tqdm
from openai import OpenAI
from neo4j import GraphDatabase

def get_data():
    db_uri = st.secrets["NEO4J_URI"]
    db_user = st.secrets["NEO4J_USERNAME"]
    db_pass = st.secrets["NEO4J_PASSWORD"]
    query_text = """
MATCH (r:Object)
RETURN r.id AS id, r.description AS text
    """
    
    driver = GraphDatabase.driver(db_uri, auth=(db_user, db_pass))
    with driver.session() as session:
        result = session.run(query_text)
        records = result.data()
    driver.close()
    return pd.DataFrame(records)

def compute_and_save_vectors():
    client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])
    model_name = "text-embedding-ada-002"
    print("Fetching records for processing.")
    df = get_data()
    print(f"Fetched {len(df)} records for processing.")

    def embed_batch(inputs, retries=2):
        for attempt in range(retries):
            try:
                result = client.embeddings.create(input=inputs, model=model_name)
                return [item.embedding for item in result.data]
            except Exception as err:
                print(f"Embedding error (attempt {attempt + 1}): {err}")
                time.sleep(2 ** attempt)
        return [None] * len(inputs)

    batch_limit = 1024
    vectors = []
    for idx in tqdm(range(0, len(df), batch_limit), desc="Generating vectors"):
        chunk = df["text"].iloc[idx:idx + batch_limit].tolist()
        vecs = embed_batch(chunk)
        vectors.extend(vecs)

    df["embedding"] = vectors
    df.to_csv("vectorized_objects.csv", index=False)
    print("Process complete. Output written to 'vectorized_questions.csv'.")

if __name__ == "__main__":
    compute_and_save_vectors()
