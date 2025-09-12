from neo4j import GraphDatabase
from neo4j_graphrag.llm import OpenAILLM
from neo4j_graphrag.embeddings import OpenAIEmbeddings
from neo4j_graphrag.retrievers import VectorRetriever
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

driver = GraphDatabase.driver(
    NEO4J_URI, 
    auth=(
        NEO4J_USER, 
        NEO4J_PASSWORD
    ))
driver.verify_connectivity()

# --- Initialize LLM and Embedder ---
llm = OpenAILLM(model_name='gpt-4o', api_key=OPENAI_API_KEY)
embedder = OpenAIEmbeddings(api_key=OPENAI_API_KEY)

# --- Initialize Vector Retriever ---
vector_retriever = VectorRetriever(
    driver=driver,
    index_name='chunkEmbeddings',
    embedder=embedder,
    return_properties=['text'])

# --- Simple Vector Search ---
query = "What are the risks that Apple faces?"
result = vector_retriever.search(query_text=query, top_k=10)
for item in result.items:
    print(f"Score: {item.metadata['score']:.4f}, Content: {item.content[0:100]}..., id: {item.metadata['id']}")

# --- Initialize RAG and Perform Search ---
query = "What companies mention AI in their filings?"
rag = GraphRAG(
    llm=llm,
    retriever=vector_retriever
)
response = rag.search(query)

print(response.answer)


"""
What are the risks that Apple faces?
What products does Microsoft reference?
What warnings have Nvidia given?
What companies mention AI in their filings?
"""