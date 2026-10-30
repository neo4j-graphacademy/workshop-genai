import os
from dotenv import load_dotenv
load_dotenv()

from neo4j import GraphDatabase
from neo4j_graphrag.llm import OpenAILLM
from neo4j_graphrag.generation import GraphRAG
from neo4j_graphrag.retrievers import Text2CypherRetriever

# Connect to Neo4j database
driver = GraphDatabase.driver(
    os.getenv("NEO4J_URI"), 
    auth=(
        os.getenv("NEO4J_USERNAME"), 
        os.getenv("NEO4J_PASSWORD")
    )
)

llm = OpenAILLM(
    model_name="gpt-5.2",
    api_key=os.getenv("OPENAI_API_KEY"),
    base_url=os.getenv("OPENAI_BASE_URL"),
)

# tag::examples[]
# Cypher examples as input/query pairs
examples = [
    "USER INPUT: 'Find a node with a nameToSearch' QUERY: MATCH (node) WHERE toLower(node.name) CONTAINS toLower(nameToSearch) RETURN node.name AS name, labels(node) AS labels",
]
# end::examples[]

# tag::retriever[]
# Build the retriever
retriever = Text2CypherRetriever(
    driver=driver,
    neo4j_database=os.getenv("NEO4J_DATABASE"),
    llm=llm,
    examples=examples,
)
# end::retriever[]

rag = GraphRAG(
    retriever=retriever, 
    llm=llm
)

query_text = "How many technologies are mentioned in the knowledge graph?"

response = rag.search(
    query_text=query_text,
    return_context=True
    )

# tag::print_response[]
print(response.answer)
print("CYPHER :", response.retriever_result.metadata["cypher"])
print("CONTEXT:", response.retriever_result.items)
# end::print_response[]

driver.close()


"""
# tag::example_queries[]
query_text = "How many technologies are mentioned in the knowledge graph?"
query_text = "How does Neo4j relate to other technologies?"
query_text = "What entities exist in the knowledge graph?" 
query_text = "What resource urls are referenced?"
# end::example_queries[]
"""
