import os
from dotenv import load_dotenv
load_dotenv()

from neo4j import GraphDatabase
# tag::import_embedder[]
from neo4j_graphrag.embeddings.openai import OpenAIEmbeddings
# end::import_embedder[]
# tag::import_retriever[]
from neo4j_graphrag.retrievers import VectorCypherRetriever
# end::import_retriever[]
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

# tag::embedder[]
# Create embedder
embedder = OpenAIEmbeddings(model="text-embedding-ada-002")
# end::embedder[]

# tag::retrieval_query[]
# Define retrieval query
retrieval_query = """
MATCH (node)-[:FROM_DOCUMENT]->(d)-[:PDF_OF]->(lesson)
RETURN
    node.text as text, score,
    lesson.url as lesson_url,
    collect { 
        MATCH (node)<-[:FROM_CHUNK]-(entity)-[r]->(other)-[:FROM_CHUNK]->()
        WITH toStringList([
            labels(entity)[2], 
            entity.name, 
            entity.type, 
            entity.description, 
            type(r), 
            labels(other)[2], 
            other.name, 
            other.type, 
            other.description
            ]) as values
        RETURN reduce(acc = "", item in values | acc || coalesce(item || ' ', ''))
    } as associated_entities
"""
# end::retrieval_query[]

# Create retriever
# tag::retriever[]
vector_retriever = VectorCypherRetriever(
    driver,
    neo4j_database=os.getenv("NEO4J_DATABASE"),
    index_name="chunkEmbedding",
    embedder=embedder,
    retrieval_query=retrieval_query,
)
# end::retriever[]

# Define functions for each tool in the agent

@tool("Get-graph-database-schema")
def get_schema():
    """Get the schema of the graph database."""
    results, summary, keys = driver.execute_query(
        "CALL db.schema.visualization()",
        database_=os.getenv("NEO4J_DATABASE")
    )
    return results

# Define a tool to retrieve lesson content
# tag::search_lessons[]
@tool("Search-lesson-content")
def search_lessons(query: str):
    """Search for lesson content related to the query."""
    # Use the vector to find relevant chunks
    result = vector_retriever.search(
        query_text=query, 
        top_k=5
    )
    context = [item.content for item in result.items]
    return context
# end::search_lessons[]


# Define a list of tools for the agent
# tag::tools[]
tools = [get_schema, search_lessons]
# end::tools[]

# Create the agent with the model and tools
agent = create_agent(
    model, 
    tools
)

# Run the application
# tag::query[]
query = "What are the benefits of using GraphRAG"
# end::query[]

for step in agent.stream(
    {
        "messages": [{"role": "user", "content": query}]
    },
    stream_mode="values",
):
    step["messages"][-1].pretty_print()


# tag::example_queries[]
"""
"What are the benefits of using GraphRAG"
"How are Knowledge Graphs associated with other technologies?"
"Summarize what concepts are associated with Knowledge Graphs?"
"How would you minimize hallucinations in LLMs?"
"""
# end::example_queries[]