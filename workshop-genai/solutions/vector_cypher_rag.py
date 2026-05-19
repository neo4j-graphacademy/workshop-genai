import os
from dotenv import load_dotenv
load_dotenv()

from neo4j import GraphDatabase
from neo4j_graphrag.embeddings.openai import OpenAIEmbeddings
from neo4j_graphrag.retrievers import VectorCypherRetriever
from neo4j_graphrag.llm import OpenAILLM
from neo4j_graphrag.generation import GraphRAG

# Connect to Neo4j database
driver = GraphDatabase.driver(
    os.getenv("NEO4J_URI"), 
    auth=(
        os.getenv("NEO4J_USERNAME"), 
        os.getenv("NEO4J_PASSWORD")
    )
)

# Create embedder
embedder = OpenAIEmbeddings(model="text-embedding-3-small")

# Define retrieval query
# tag::simple_retrieval_query[]
retrieval_query = """
RETURN node.text as text, score
"""
# end::simple_retrieval_query[]
# tag::retrieval_query[]
retrieval_query = """
MATCH (node)-[:FROM_DOCUMENT]->(d)-[:PDF_OF]->(lesson)
RETURN DISTINCT
    node.text as text, score,
    lesson.url as lesson_url,
    collect { MATCH (node)<-[:FROM_CHUNK]-(e:Technology) RETURN e.name } as technologies,
    collect { MATCH (node)<-[:FROM_CHUNK]-(e:Concept) RETURN e.name } as concepts
"""
# end::retrieval_query[]
# tag::advanced_retrieval_query[]
retrieval_query = """
MATCH (node)-[:FROM_DOCUMENT]->(d)-[:PDF_OF]->(lesson)
RETURN
    node.text as text, score,
    lesson.url as lesson_url,
    collect { 
        MATCH (node)<-[:FROM_CHUNK]-(entity)-[r]->(other)-[:FROM_CHUNK]->()
        WITH toStringList([
            [l IN labels(entity)
                WHERE NOT l IN ["__KGBuilder__", "__Entity__"]][0],
            entity.name, 
            type(r), 
            [l IN labels(other)
                WHERE NOT l IN ["__KGBuilder__", "__Entity__"]][0],
            other.name 
            ]) as values
        RETURN reduce(acc = "", item in values | acc || coalesce(item || ' ', ''))
    } as associated_entities
"""
# end::advanced_retrieval_query[]

# Create retriever
# tag::retriever[]
retriever = VectorCypherRetriever(
    driver,
    neo4j_database=os.getenv("NEO4J_DATABASE"),
    index_name="chunkEmbedding",
    embedder=embedder,
    retrieval_query=retrieval_query,
)
# end::retriever[]

#  Create the LLM
llm = OpenAILLM(
    model_name="gpt-5.2"
)

# Create GraphRAG pipeline
rag = GraphRAG(retriever=retriever, llm=llm)

# Search
query_text = "What is a knowledge graphs and where can I learn about them?"

response = rag.search(
    query_text=query_text, 
    retriever_config={"top_k": 5},
    return_context=True
)

print(response.answer)
# tag::print_context[]
print("CONTEXT:", response.retriever_result.items)
# end::print_context[]

# Close the database connection
driver.close()


"""
# tag::example_queries[]
query_text = "What is a knowledge graphs and where can I learn about them?"
query_text = "What technologies and concepts support knowledge graphs?"
query_text = "Describe Retrieval Augmented Generation."
# end::example_queries[]
"""
