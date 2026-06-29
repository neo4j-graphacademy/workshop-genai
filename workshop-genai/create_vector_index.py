import os

from dotenv import load_dotenv
from neo4j import GraphDatabase

load_dotenv()

VECTOR_INDEX_CYPHER = """
CREATE VECTOR INDEX chunkEmbedding IF NOT EXISTS
FOR (n:Chunk)
ON n.embedding
OPTIONS {indexConfig: {
  `vector.dimensions`: 1536,
  `vector.similarity_function`: 'cosine'
}}
"""


def main():
    driver = GraphDatabase.driver(
        os.getenv("NEO4J_URI"),
        auth=(os.getenv("NEO4J_USERNAME"), os.getenv("NEO4J_PASSWORD")),
    )

    try:
        driver.verify_connectivity()
        driver.execute_query(
            VECTOR_INDEX_CYPHER,
            database_=os.getenv("NEO4J_DATABASE"),
        )
        print("Vector index `chunkEmbedding` is ready.")
    finally:
        driver.close()


if __name__ == "__main__":
    main()
