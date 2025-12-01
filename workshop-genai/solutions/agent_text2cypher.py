import os
from dotenv import load_dotenv
load_dotenv()

from neo4j import GraphDatabase
from neo4j_graphrag.embeddings.openai import OpenAIEmbeddings
from neo4j_graphrag.retrievers import VectorCypherRetriever
# tag::import_llm[]
from neo4j_graphrag.llm import OpenAILLM
# end::import_llm[]
# tag::import_retriever[]
from neo4j_graphrag.retrievers import Text2CypherRetriever
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

# Create embedder
embedder = OpenAIEmbeddings(model="text-embedding-ada-002")

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

# Create vector retriever
vector_retriever = VectorCypherRetriever(
    driver,
    neo4j_database=os.getenv("NEO4J_DATABASE"),
    index_name="chunkEmbedding",
    embedder=embedder,
    retrieval_query=retrieval_query,
)

# tag::llm[]
# Crete LLM for Text2CypherRetriever
llm = OpenAILLM(
    model_name="gpt-4o", 
    model_params={"temperature": 0}
)
# end::llm[]

# tag::retriever[]
# Cypher examples as input/query pairs
examples = [
    "USER INPUT: 'Find a node with the name $name?' QUERY: MATCH (node) WHERE toLower(node.name) CONTAINS toLower($name) RETURN node.name AS name, labels(node) AS labels",
]

# Build the retriever
text2cypher_retriever = Text2CypherRetriever(
    driver=driver,
    neo4j_database=os.getenv("NEO4J_DATABASE"),
    llm=llm,
    examples=examples,
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

# tag::query_database[]
# Define a tool to query the database
@tool("Query-database")
def query_database(query: str):
    """A catchall tool to get answers to specific questions about lesson content."""
    result = text2cypher_retriever.get_search_results(query)
    return result
# end::query_database[]



# Define a list of tools for the agent
# tag::tools[]
tools = [get_schema, search_lessons, query_database]
# end::tools[]

# Create the agent with the model and tools
agent = create_agent(
    model, 
    tools
)

# Run the application
# tag::query[]
query = "How many lessons are there?"
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
How many lessons are there?
Each lesson is part of a module. How many lessons are in each module?
Search the graph and return a list of challenges.
What benefits are associated to the technologies described in the knowledge graph?
"""
# end::example_queries[]    