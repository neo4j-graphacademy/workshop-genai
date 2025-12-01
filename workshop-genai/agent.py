import os
from dotenv import load_dotenv
load_dotenv()

from neo4j import GraphDatabase
from langchain.chat_models import init_chat_model
from langchain.agents import create_agent
from langchain_core.tools import tool

# Initialize the chat model
model = init_chat_model("gpt-4o", model_provider="openai")

# Connect to Neo4j database
driver = GraphDatabase.driver(
    os.getenv("NEO4J_URI"), 
    auth=(
        os.getenv("NEO4J_USERNAME"), 
        os.getenv("NEO4J_PASSWORD")
    )
)

# Define functions for each tool in the agent

@tool("Get-graph-database-schema")
def get_schema():
    """Get the schema of the graph database."""
    results, summary, keys = driver.execute_query(
        "CALL db.schema.visualization()",
        database_=os.getenv("NEO4J_DATABASE")
    )
    return results

# Define a list of tools for the agent
tools = [get_schema]

# Create the agent with the model and tools
agent = create_agent(
    model, 
    tools
)

# Run the application
query = "Summarise the schema of the graph database."

for step in agent.stream(
    {
        "messages": [{"role": "user", "content": query}]
    },
    stream_mode="values",
):
    step["messages"][-1].pretty_print()

