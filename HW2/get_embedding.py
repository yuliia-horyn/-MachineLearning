from neo4j import GraphDatabase
import streamlit as st  # comment out if not using Streamlit

# Neo4j connection credentials
uri = st.secrets["NEO4J_URI"]
username = st.secrets["NEO4J_USERNAME"]
password = st.secrets["NEO4J_PASSWORD"]


load_embeddings_query = """
LOAD CSV WITH HEADERS
FROM 'https://www.dropbox.com/scl/fi/415v3o0pmbz2xim7owe4q/vectorized_objects.csv?rlkey=gl9866tln7jklkrndll7tk38z&st=mtjneciy&dl=1'
AS row
MATCH (o:Object {id: row.id})
CALL db.create.setVectorProperty(o, 'textEmbedding', apoc.convert.fromJsonList(row.embedding))
YIELD node
RETURN count(*)
"""

drop_index = """
DROP INDEX object_text_embedding_index IF EXISTS
"""
create_vector_index_query = """

CREATE VECTOR INDEX object_text_embedding_index IF NOT EXISTS
FOR (o:Object)
ON o.textEmbedding
OPTIONS {
  indexConfig: {
    `vector.dimensions`: 1536,
    `vector.similarity_function`: 'cosine'
  }
}
"""

def exec_query(driver, query: str):
    with driver.session() as session:
        result = session.run(query)
        try:
            return result.data()
        except Exception as e:
            st.error(f"Failed to retrieve results: {e}")
            return []

if __name__ == "__main__":
    driver = GraphDatabase.driver(uri, auth=(username, password))

    st.write("Running embedding load query...")
    result1 = exec_query(driver, load_embeddings_query)
    st.success(f"Embeddings loaded: {result1}")
    exec_query(driver, drop_index)
    st.write("Creating vector index...")
    result2 = exec_query(driver, create_vector_index_query)
    driver.close()
