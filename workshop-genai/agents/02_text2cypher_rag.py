import os
from dotenv import load_dotenv
load_dotenv()

from langchain_core.documents import Document
from langchain.chat_models import init_chat_model
from langgraph.graph import START, StateGraph
from langchain_core.prompts import PromptTemplate
from typing_extensions import List, TypedDict
from langchain_neo4j import Neo4jGraph
# tag::import_cypher_qa[]
from langchain_neo4j import GraphCypherQAChain
# end::import_cypher_qa[]

# Initialize the LLM
model = init_chat_model("gpt-4o", model_provider="openai")

# tag::cypher_model[]
cypher_model = init_chat_model(
    "gpt-4o", 
    model_provider="openai",
    temperature=0.0
)
# end::cypher_model[]

# Create a prompt
template = """Use the following pieces of context to answer the question at the end.
If you don't know the answer, just say that you don't know, don't try to make up an answer.

{context}

Question: {question}

Answer:"""

prompt = PromptTemplate.from_template(template)

# Define state for application
class State(TypedDict):
    question: str
    context: List[dict]
    answer: str

# Connect to Neo4j
graph = Neo4jGraph(
    url=os.getenv("NEO4J_URI"),
    username=os.getenv("NEO4J_USERNAME"), 
    password=os.getenv("NEO4J_PASSWORD"),
)

# Create a cypher generation prompt
cypher_template = """Task:Generate Cypher statement to query a graph database.
Instructions:
Use only the provided relationship types and properties in the schema.
Do not use any other relationship types or properties that are not provided.

Use `WHERE tolower(node.name) CONTAINS toLower('name')` to filter nodes by name.

Schema:
{schema}

Note: Do not include any explanations or apologies in your responses.
Do not respond to any questions that might ask anything else than for you to construct a Cypher statement.
Do not include any text except the generated Cypher statement.

The question is:
{question}"""

cypher_prompt = PromptTemplate(
    input_variables=["schema", "question"], 
    template=cypher_template
)

# tag::cypher_qa[]
# Create the Cypher QA chain
cypher_qa = GraphCypherQAChain.from_llm(
    graph=graph, 
    llm=model,
    cypher_llm=cypher_model,
    cypher_prompt=cypher_prompt,
    allow_dangerous_requests=True,
    return_direct=True,
    verbose=True
)
# end::cypher_qa[]

# Define functions for each step in the application

# tag::retrieve[]
# Retrieve context 
def retrieve(state: State):
    context = cypher_qa.invoke(
        {"query": state["question"]}
    )
    return {"context": context}
# end::retrieve[]

# Generate the answer based on the question and context
def generate(state: State):
    messages = prompt.invoke({"question": state["question"], "context": state["context"]})
    response = model.invoke(messages)
    return {"answer": response.content}

# Define application steps
workflow = StateGraph(State).add_sequence([retrieve, generate])
workflow.add_edge(START, "retrieve")
app = workflow.compile()

# Run the application
question = "What financial metrics does Apple have"
response = app.invoke({"question": question})
print("Answer:", response["answer"])
print("Context:", response["context"])

"""
what are the company names of companies owned by BlackRock Inc.
What financial metrics does APPLE INC have?
What stock has MICROSOFT CORP issued?
Summarise what products are mentioned regarding the NVIDIA CORPORATION?
"""