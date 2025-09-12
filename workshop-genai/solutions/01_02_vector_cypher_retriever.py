from neo4j import GraphDatabase
from neo4j_graphrag.llm import OpenAILLM
from neo4j_graphrag.embeddings import OpenAIEmbeddings
from neo4j_graphrag.retrievers import VectorCypherRetriever
from neo4j_graphrag.generation import GraphRAG

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

# --- VectorCypherRetriever Example: Detailed Search with Context 
detail_context_query = """
MATCH (node)-[:FROM_DOCUMENT]-(doc:Document)-[:FILED]-(company:Company)-[:FACES_RISK]->(risk:RiskFactor)
RETURN company.name AS company, collect(DISTINCT risk.name) AS risks, node.text AS context
"""

vector_cypher_retriever = VectorCypherRetriever(
    driver=driver,
    index_name='chunkEmbeddings',
    embedder=embedder,
    retrieval_query=detail_context_query
)

# --- Initialize RAG and Perform Search ---
query = "What are the top risk factors that Apple faces?"

rag = GraphRAG(llm=llm, retriever=vector_cypher_retriever)
response = rag.search(query)
print(response.answer)

# --- Initialize RAG, search with options and return context ---
rag = GraphRAG(llm=llm, retriever=vector_cypher_retriever)
response = rag.search(
    query,
    retriever_config={"top_k": 5},
    return_context=True
    )
print(response.answer)
print("Context:", *response.retriever_result.items, sep="\n\n")

## VectorCypherRetriever Example: Detailed Search with Context
asset_manager_query = """
MATCH (node)-[:FROM_DOCUMENT]-(doc:Document)-[:FILED]-(company:Company)-[:OWNS]-(manager:AssetManager)
RETURN company.name AS company, manager.managerName AS AssetManagerWithSharesInCompany, node.text AS context
"""

vector_cypher_retriever = VectorCypherRetriever(
    driver=driver,
    index_name='chunkEmbeddings',
    embedder=embedder,
    retrieval_query=asset_manager_query
)

query = "Who are the asset managers most affected by banking regulations?"
rag = GraphRAG(llm=llm, retriever=vector_cypher_retriever)
response = rag.search(
    query,
    retriever_config={"top_k": 5},
    return_context=True
    )
print(response.answer)
# print("Context:", *response.retriever_result.items, sep="\n\n")


## VectorCypherRetriever Example: Finding Shared Risks Among Companies
vector_company_risk_query = """
WITH node
MATCH (node)-[:FROM_DOCUMENT]-(doc:Document)-[:FILED]-(c1:Company)
MATCH (c1)-[:FACES_RISK]->(risk:RiskFactor)<-[:FACES_RISK]-(c2:Company)
WHERE c1 <> c2
RETURN
  c1.name AS source_company,
  collect(DISTINCT c2.name) AS related_companies,
  collect(DISTINCT risk.name) AS shared_risks
LIMIT 10
"""

vector_cypher_retriever = VectorCypherRetriever(
    driver=driver,
    index_name="chunkEmbeddings",
    embedder=embedder,
    retrieval_query=vector_company_risk_query
)

query = "What risks connect major tech companies?"
rag = GraphRAG(llm=llm, retriever=vector_cypher_retriever)
response = rag.search(
    query,
    retriever_config={"top_k": 5},
    return_context=True
    )
print(response.answer)
# print("Context:", *response.retriever_result.items, sep="\n\n")