from neo4j import GraphDatabase
from neo4j_graphrag.llm import OpenAILLM
from neo4j_graphrag.embeddings import OpenAIEmbeddings
from neo4j_graphrag.retrievers import Text2CypherRetriever
from neo4j_graphrag.generation import GraphRAG
from neo4j_graphrag.schema import get_schema
from neo4j_graphrag.experimental.components.schema import SchemaFromExistingGraphExtractor
from neo4j_graphrag.experimental.utils.schema import schema_visualization
import asyncio

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

async def visualize_schema():
    """
    Create an interactive HTML visualization of the graph schema.

    This function:
    1. Extracts the structured schema from the Neo4j database
    2. Creates an interactive visualization using neo4j-viz
    3. Saves the visualization as an HTML file

    The visualization shows:
    - All node types (labels) as nodes
    - All relationship types as connections between nodes
    - Properties for each node and relationship type
    """
    # Extract structured schema from the existing graph
    # This queries the database to get all node types, relationship types, and properties
    schema_extractor = SchemaFromExistingGraphExtractor(driver=driver)
    graph_schema = await schema_extractor.run()

    # Create the visualization
    # VG is a VisualizationGraph object that contains nodes and relationships
    VG = schema_visualization(graph_schema)
    html = VG.render()

    # Save the generated HTML file
    # This creates an interactive graph that can be opened in a web browser
    with open("my_schema.html", "w") as f:
        f.write(html.data)
    print("Schema visualization saved to my_schema.html")

# Run the async function to generate the visualization
asyncio.run(visualize_schema())

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

# Close the driver connection to prevent cleanup warnings
driver.close()

"""
Summarise the products mentioned in the company filings.
"""
