import os
from dotenv import load_dotenv
load_dotenv()

from langchain.chat_models import init_chat_model
from langgraph.prebuilt import create_react_agent
from langchain_core.tools import tool
from langchain_neo4j import Neo4jGraph

# Initialize the LLM
model = init_chat_model("gpt-4o", model_provider="openai")

# Connect to Neo4j
graph = Neo4jGraph(
    url=os.getenv("NEO4J_URI"),
    username=os.getenv("NEO4J_USERNAME"), 
    password=os.getenv("NEO4J_PASSWORD"),
)

# Define functions for each tool in the agent

@tool("Get-graph-database-schema")
def get_schema():
    """Get the schema of the graph database."""
    context = graph.schema
    return context

# Define a list of tools for the agent
tools = [get_schema]

# Create the agent with the model and tools
agent = create_react_agent(
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



"""
Summarise the schema of the graph database.
What questions can I answer using this graph database?
How are Products related to other entities?
How does the graph model relate financial documents to risk factors?
"""