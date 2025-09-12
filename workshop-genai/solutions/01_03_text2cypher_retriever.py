from neo4j import GraphDatabase
from neo4j_graphrag.llm import OpenAILLM
from neo4j_graphrag.embeddings import OpenAIEmbeddings
from neo4j_graphrag.retrievers import Text2CypherRetriever
from neo4j_graphrag.generation import GraphRAG
from neo4j_graphrag.schema import get_schema

# Load environment variables
import os
from dotenv import load_dotenv
load_dotenv()
NEO4J_URI = os.getenv('NEO4J_URI')
NEO4J_USER = os.getenv('NEO4J_USERNAME')
NEO4J_PASSWORD = os.getenv('NEO4J_PASSWORD')
NEO4J_DATABASE = os.getenv('NEO4J_DATABASE')
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')

driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))

# --- Initialize LLM and Embedder ---
llm = OpenAILLM(model_name='gpt-4o', api_key=OPENAI_API_KEY)
embedder = OpenAIEmbeddings(api_key=OPENAI_API_KEY)

schema = get_schema(driver)
print(schema)

# --- Text2CypherRetriever Example ---
text2cypher_retriever = Text2CypherRetriever(
    driver=driver,
    llm=llm,
    neo4j_schema=schema
)

query = "What companies are owned by BlackRock Inc."
cypher_query = text2cypher_retriever.get_search_results(query)

print("Original Query:", query)
print("Generated Cypher:", cypher_query.metadata["cypher"])

print("Cypher Query Results:")
for record in cypher_query.records:
    print(record)
    

# --- Initialize RAG and Perform Search ---
query = "Who are the assets managers?"
rag = GraphRAG(llm=llm, retriever=text2cypher_retriever)
response = rag.search(
    query,
    return_context=True
    )
print(response.answer)
print("Generate Cypher:", response.retriever_result.metadata["cypher"])
print("Context:", *response.retriever_result.items, sep="\n")


"""
Summarise the products mentioned in the company filings.
"""