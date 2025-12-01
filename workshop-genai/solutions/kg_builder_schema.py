import os
from dotenv import load_dotenv
load_dotenv()

import asyncio

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

# tag::simple_nodes[]
NODE_TYPES = [
    "Technology",
    "Concept",
    "Example",
    "Process",
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
    },
)
# end::simple_nodes[]

# tag::node_types[]
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
# end::node_types[]

# tag::relationship_types[]
RELATIONSHIP_TYPES = [
    "RELATED_TO",
    "PART_OF",
    "USED_IN",
    "LEADS_TO",
    "HAS_CHALLENGE",
    "LEADS_TO",
    "CITES"
]
# end::relationship_types[]

# tag::patterns[]
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
# end::patterns[]

# tag::kg_builder[]
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
# end::kg_builder[]

pdf_file = "./workshop-genai/data/genai-fundamentals_1-generative-ai_1-what-is-genai.pdf"
result = asyncio.run(kg_builder.run_async(file_path=pdf_file))
print(result.result)

# tag::all_documents[]
data_path = "./workshop-genai/data/"
pdf_files = [os.path.join(data_path, f) for f in os.listdir(data_path) if f.endswith('.pdf')]

for pdf_file in pdf_files:

    print(f"Processing {pdf_file}")
    result = asyncio.run(kg_builder.run_async(file_path=pdf_file))
    print(result.result)
# end::all_documents[]