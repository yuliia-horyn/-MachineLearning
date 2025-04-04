import streamlit as st
from llm import llm
from graph import graph
from langchain.schema import StrOutputParser
from operator import itemgetter
from langchain_neo4j import GraphCypherQAChain
from langchain_core.runnables import RunnableLambda, RunnableMap


from langchain.prompts.prompt import PromptTemplate

CYPHER_GENERATION_TEMPLATE = """
You are an expert Neo4j Developer translating user questions into Cypher to answer questions on crime scenes.
Convert the user's question based on the schema.

Use only the provided relationship types and properties in the schema.
Do not use any other relationship types or properties that are not provided.

Do not return entire nodes or embedding properties.


Examples:
Show me 25 crime scenes
MATCH (n:Location) RETURN n LIMIT 25


Crimes investigated by Inspector Morse
MATCH (o:Officer {rank: 'Chief Inspector', surname: 'Morse'})<-[i:INVESTIGATED_BY]-(c:Crime)
RETURN *


Crimes near a particular address
MATCH (l:Location {address: '1 Coronation Street', postcode: 'M5 3RW'})
WITH point(l) AS corrie
MATCH (x:Location)-[:HAS_POSTCODE]->(p:PostCode),
(x)<-[:OCCURRED_AT]-(c:Crime)
WITH x, p, c, point.distance(point(x), corrie) AS distance
WHERE distance < 500
RETURN x.address AS address, p.code AS postcode, count(c) AS crime_total, collect(distinct(c.type)) AS crime_type, distance
ORDER BY distance
LIMIT 10


Schema:
{schema}

Question:
{question}

Cypher Query:
"""

cypher_prompt = PromptTemplate.from_template(CYPHER_GENERATION_TEMPLATE)



cypher_qa = GraphCypherQAChain.from_llm(
    llm,
    graph=graph,
    verbose=True,
    allow_dangerous_requests=True,
    cypher_prompt=cypher_prompt,

)

get_crime = (
    cypher_qa 
    | RunnableLambda(lambda output: "For query:" + output.get('query') + "we found objects: " + output.get('result', "None"))

)
