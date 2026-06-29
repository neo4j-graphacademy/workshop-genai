# Aura Setup

This repository accompanies the [Neo4j and Generative AI Workshop](https://graphacademy.neo4j.com/courses/workshop-genai) on [GraphAcademy](https://graphacademy.neo4j.com).

When the devcontainer is created, such as in a GitHub codespace, all the required software and packages will be installed.

This edition uses Neo4j AuraDB Free instead of Neo4j Sandbox.
Each workshop participant should create and use their own AuraDB Free instance.
Do not point multiple participants at the same database because the workshop creates graph data, embeddings, and indexes.

To get started:

1. Create or sign in to a Neo4j Aura account at [Aura Console](https://console.neo4j.io/).
2. Create a new AuraDB instance and choose the Free tier. AuraDB Free is limited to one Free instance per account, so each participant should use their own account/instance.
3. Download or copy the credentials for your new AuraDB Free instance.
4. Create a new [`.env`](../.env) file and copy the contents of [`.env.example`](../.env.example) into it.
5. Update the Neo4j values with the values from your own Aura credentials file.
6. Add your `OPENAI_API_KEY`.
7. Run [`test_environment.py`](../test_environment.py) to check the environment is set up correctly.
8. Run [`kg_structured_builder.py`](../workshop-genai/kg_structured_builder.py) or the completed [`solutions/kg_structured_builder.py`](../workshop-genai/solutions/kg_structured_builder.py) to build the graph.
9. Run [`create_vector_index.py`](../workshop-genai/create_vector_index.py) before the vector RAG examples.
10. Choose a path:
    - Python GraphRAG examples in [`workshop-genai`](../workshop-genai).
    - Managed Aura Agent setup in [`aura-agent`](../aura-agent/README.adoc).

Use the Aura driver URI for `NEO4J_URI`.
It usually starts with `neo4j+s://`, for example:

```env
NEO4J_URI="neo4j+s://<your-instance-id>.databases.neo4j.io"
NEO4J_USERNAME="neo4j"
NEO4J_PASSWORD="<your-aura-password>"
NEO4J_DATABASE="neo4j"
```
