import os
from dotenv import load_dotenv
load_dotenv()

from langchain.chat_models import init_chat_model
from langgraph.prebuilt import create_react_agent
from langchain_core.tools import tool
from langchain_neo4j import Neo4jGraph, Neo4jVector, GraphCypherQAChain
from langchain_openai import OpenAIEmbeddings
from langchain_core.prompts import PromptTemplate

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

# Create a separate model for Cypher generation
cypher_model = init_chat_model(
    "gpt-4o", 
    model_provider="openai",
    temperature=0.0
)

# Create a cypher generation prompt
cypher_template = """Task:Generate Cypher statement to query a graph database.
Instructions:
Use only the provided relationship types and properties in the schema.
Do not use any other relationship types or properties that are not provided.

Use `WHERE tolower(node.name) CONTAINS toLower('name')` to filter nodes by name.

Schema:
{schema}

Note: Do not include any explanations or apologies in your responses.
Do not respond to any questions that might ask anything else than for you to construct a Cypher statement.
Do not include any text except the generated Cypher statement.

The question is:
{question}"""

cypher_prompt = PromptTemplate(
    input_variables=["schema", "question"], 
    template=cypher_template
)

# Create the Cypher QA chain
cypher_qa = GraphCypherQAChain.from_llm(
    graph=graph, 
    llm=model,
    cypher_llm=cypher_model,
    cypher_prompt=cypher_prompt,
    allow_dangerous_requests=True,
    return_direct=True,
    verbose=True
)

# Define functions for each tool in the agent

@tool("Get-graph-database-schema")
def get_schema():
    """Get the schema of the graph database."""
    context = graph.schema
    return context

@tool("Retrieve-financial-documents")
def retrieve_docs(query: str):
    """Find details about companies in their financial documents."""
    # Use the vector to find relevant documents
    context = chunk_vector.similarity_search(
        query, 
        k=3,
    )
    return context

# Define a tool to query the database
@tool("Query-database")
def query_database(query: str):
    """Get answers to specific questions about companies, risks, and financial metrics."""
    context = cypher_qa.invoke(
        {"query": query}
    )
    return {"context": context}

# Add the tools to the agent
tools = [get_schema, retrieve_docs, query_database]

agent = create_react_agent(
    model, 
    tools
)

# Run the application
query = "What stock has Microsoft issued?"

for step in agent.stream(
    {
        "messages": [{"role": "user", "content": query}]
    },
    stream_mode="values",
):
    step["messages"][-1].pretty_print()



"""
What stock has Microsoft issued?
How does the graph model relate to financial documents and risk factors?
What are the top risk factors that Apple faces?
Summarize what risk factors are mentioned in Apple's financial documents?
What are the top risk factors that Apple faces?
How many risk facts does Apple face and what are the top ones?
"""