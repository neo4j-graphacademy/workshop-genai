import os
from dotenv import load_dotenv
load_dotenv()

import asyncio
import csv

from neo4j import GraphDatabase
from neo4j_graphrag.llm import OpenAILLM
from neo4j_graphrag.embeddings import OpenAIEmbeddings
from neo4j_graphrag.experimental.pipeline.kg_builder import SimpleKGPipeline
from neo4j_graphrag.experimental.components.text_splitters.fixed_size_splitter import FixedSizeSplitter

neo4j_driver = GraphDatabase.driver(
    os.getenv("NEO4J_URI"),
    auth=(os.getenv("NEO4J_USERNAME"), os.getenv("NEO4J_PASSWORD"))
)
neo4j_driver.verify_connectivity()

llm = OpenAILLM(
    model_name="gpt-4o",
    model_params={
        "temperature": 0,
        "response_format": {"type": "json_object"},
    }
)

embedder = OpenAIEmbeddings(
    model="text-embedding-ada-002"
)

text_splitter = FixedSizeSplitter(chunk_size=500, chunk_overlap=100)

NODE_TYPES = [
    "Technology",
    "Concept",
    "Example",
    "Process",
    "Challenge",
    {"label": "Benefit", "description": "A benefit or advantage of using a technology or approach."},
    {
        "label": "Resource",
        "description": "A related learning resource such as a book, article, video, or course.",
        "properties": [
            {"name": "name", "type": "STRING", "required": True}, 
            {"name": "type", "type": "STRING"}
        ]
    },
]

RELATIONSHIP_TYPES = [
    "RELATED_TO",
    "PART_OF",
    "USED_IN",
    "LEADS_TO",
    "HAS_CHALLENGE",
    "LEADS_TO",
    "CITES"
]

PATTERNS = [
    ("Technology", "RELATED_TO", "Technology"),
    ("Concept", "RELATED_TO", "Technology"),
    ("Example", "USED_IN", "Technology"),
    ("Process", "PART_OF", "Technology"),
    ("Technology", "HAS_CHALLENGE", "Challenge"),
    ("Concept", "HAS_CHALLENGE", "Challenge"),
    ("Technology", "LEADS_TO", "Benefit"),
    ("Process", "LEADS_TO", "Benefit"),
    ("Resource", "CITES", "Technology"),
]

kg_builder = SimpleKGPipeline(
    llm=llm,
    driver=neo4j_driver, 
    neo4j_database=os.getenv("NEO4J_DATABASE"), 
    embedder=embedder, 
    from_pdf=True,
    text_splitter=text_splitter,
    schema={
        "node_types": NODE_TYPES,
        "relationship_types": RELATIONSHIP_TYPES,
        "patterns": PATTERNS
    },
)

# tag::load_csv[]
data_path = "./workshop-genai/data/"

docs_csv = csv.DictReader(
    open(os.path.join(data_path, "docs.csv"), encoding="utf8", newline='')
)
# end::load_csv[]

# tag::cypher[]
cypher = """
MATCH (d:Document {path: $pdf_path})
MERGE (l:Lesson {url: $url})
SET l.name = $lesson,
    l.module = $module,
    l.course = $course
MERGE (d)-[:PDF_OF]->(l)
"""
# end::cypher[]

for doc in docs_csv:

    # tag::pdf_path[]
    # Create the complete PDF path
    doc["pdf_path"] = os.path.join(data_path, doc["filename"])
    print(f"Processing document: {doc['pdf_path']}")
    # end::pdf_path[]

    # Entity extraction and KG population
    result = asyncio.run(
        kg_builder.run_async(
            file_path=os.path.join(doc["pdf_path"])
        )
    )

    # tag::create_structured_graph[]
    # Create structured graph
    records, summary, keys = neo4j_driver.execute_query(
        cypher,
        parameters_=doc,
        database_=os.getenv("NEO4J_DATABASE")
    )
    # end::create_structured_graph[]
    print(result, summary.counters)

