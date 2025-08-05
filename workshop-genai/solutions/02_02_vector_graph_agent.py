import os
from dotenv import load_dotenv
load_dotenv()

from langchain.chat_models import init_chat_model
from langgraph.prebuilt import create_react_agent
from langchain_core.tools import tool
from langchain_neo4j import Neo4jGraph, Neo4jVector
from langchain_openai import OpenAIEmbeddings


# Initialize the LLM
model = init_chat_model("gpt-4o", model_provider="openai")

# Create the embedding model
embedding_model = OpenAIEmbeddings(model="text-embedding-ada-002")

# Connect to Neo4j
graph = Neo4jGraph(
    url=os.getenv("NEO4J_URI"),
    username=os.getenv("NEO4J_USERNAME"), 
    password=os.getenv("NEO4J_PASSWORD"),
)

# Define the retrieval query
retrieval_query = """
MATCH (node)-[:FROM_DOCUMENT]-(doc:Document)-[:FILED]-(company:Company)
RETURN 
    node.text as text,
    score,
    {
        company: company.name,
        risks: [ (company:Company)-[:FACES_RISK]->(risk:RiskFactor) | risk.name ]
    } AS metadata
ORDER BY score DESC
"""

# Create Vector
chunk_vector = Neo4jVector.from_existing_index(
    embedding_model,
    graph=graph,
    index_name="chunkEmbeddings",
    embedding_node_property="embedding",
    text_node_property="text",
    retrieval_query=retrieval_query,
)

# Define functions for each tool in the agent

@tool("Get-graph-database-schema")
def get_schema():
    """Get the schema of the graph database."""
    context = graph.schema
    return context

# Define a tool to retrieve financial documents
@tool("Retrieve-financial-documents")
def retrieve_docs(query: str):
    """Find details about companies in their financial documents."""
    # Use the vector to find relevant documents
    context = chunk_vector.similarity_search(
        query, 
        k=3,
    )
    return context

# Add the tools to the agent
tools = [get_schema, retrieve_docs]

agent = create_react_agent(
    model, 
    tools
)

# Run the application
query = "What products does Microsoft referred to in its financial documents?"

for step in agent.stream(
    {
        "messages": [{"role": "user", "content": query}]
    },
    stream_mode="values",
):
    step["messages"][-1].pretty_print()



"""
Summarize what risk factors are mentioned in Apple's financial documents?
What products does Microsoft referred to in its financial documents?
What type of questions can I ask about Apple using the graph database?
"""