import streamlit as st
from llm import llm, embeddings
from graph import graph
from langchain.schema import StrOutputParser

from langchain_neo4j import Neo4jVector
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableLambda, RunnableMap


neo4jvector = Neo4jVector.from_existing_index(
    embeddings,
    graph=graph,
    index_name="object_text_embedding_index",
    node_label="Object",
    text_node_property="description",
    embedding_node_property="textEmbedding",
    retrieval_query="""
    OPTIONAL MATCH (node)-[:INVOLVED_IN]->(case:Crime)
    OPTIONAL MATCH (node)-[:TAGGED]->(tag:EvidenceTag)
    OPTIONAL MATCH (node)<-[:COLLECTED]-(officer:Officer)
    WITH node, score, case, officer, collect(DISTINCT tag.name) AS tags
    RETURN
      node.description AS text,
      score,
      {
        objectId: node.id,
        objectType: node.type,
        tags: tags,
        caseId: case.case_id,
        officerName: officer.name,
        source: 'https://internal-system.local/object/' + node.id
      } AS metadata
    """
)



retriever = neo4jvector.as_retriever()

instructions = (
    "Use the given context to answer the question."
    "If you don't know the answer, say you don't know."
    "Context: {context}"
)

prompt = ChatPromptTemplate.from_messages(
    [
        ("system", instructions),
        ("human", "{input}"),
    ]
)

question_answer_chain = create_stuff_documents_chain(llm, prompt)
crime_context_retriever = create_retrieval_chain(
    retriever, 
    question_answer_chain
)

def search_similar_question(question):
    """Searching for similar questions"""
    print("Executing embeddings search")
    return (crime_context_retriever.invoke({"input": question}) )
            # | RunnableLambda(lambda output: print(output.keys()))
            # | RunnableLambda(lambda output: output['result']))
