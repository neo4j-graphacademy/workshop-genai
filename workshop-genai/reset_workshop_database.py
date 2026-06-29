import os

from dotenv import load_dotenv
from neo4j import GraphDatabase

load_dotenv()

CONFIRMATION_TEXT = "DELETE WORKSHOP DATA"


def main():
    database = os.getenv("NEO4J_DATABASE")

    print("This will delete every node and relationship in the configured Neo4j database.")
    print("Only use this with your own AuraDB Free workshop instance.")
    confirmation = input(f"Type {CONFIRMATION_TEXT} to continue: ")

    if confirmation != CONFIRMATION_TEXT:
        print("Reset cancelled.")
        return

    driver = GraphDatabase.driver(
        os.getenv("NEO4J_URI"),
        auth=(os.getenv("NEO4J_USERNAME"), os.getenv("NEO4J_PASSWORD")),
    )

    try:
        driver.verify_connectivity()
        _, delete_summary, _ = driver.execute_query(
            "MATCH (n) DETACH DELETE n",
            database_=database,
        )

        driver.execute_query(
            "DROP INDEX chunkEmbedding IF EXISTS",
            database_=database,
        )

        counters = delete_summary.counters
        print(
            f"Deleted {counters.nodes_deleted} nodes and "
            f"{counters.relationships_deleted} relationships."
        )
        print("Dropped vector index `chunkEmbedding` if it existed.")
    finally:
        driver.close()


if __name__ == "__main__":
    main()
