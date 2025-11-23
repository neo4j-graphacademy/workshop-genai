// This Cypher script loads the data from the CSV files into a Neo4j.

// The CSV files were exported from Neo4j using `apoc.export.csv.all`.
// CALL apoc.export.csv.all("genai-data.csv", {bulkImport: true})

MATCH (n) SET n:__index__;

CREATE INDEX import__ID IF NOT EXISTS
FOR (n:__index__) ON (n.`:ID`);

CREATE VECTOR INDEX chunkEmbeddings IF NOT EXISTS
FOR (n:Chunk) ON (n.embedding)
OPTIONS { indexConfig : {
  `vector.dimensions`: 768,
  `vector.similarity_function`: 'cosine'
  }
};

CREATE CONSTRAINT managerName_AssetManager_uniq IF NOT EXISTS
FOR (n:AssetManager) REQUIRE n.managerName IS UNIQUE;

CREATE CONSTRAINT name_Company_uniq IF NOT EXISTS
FOR (n:Company) REQUIRE n.name IS UNIQUE;

CREATE CONSTRAINT path_Document_uniq IF NOT EXISTS
FOR (n:Document) REQUIRE n.path IS UNIQUE;

LOAD CSV WITH HEADERS FROM "https://raw.githubusercontent.com/neo4j-graphacademy/workshop-genai/refs/heads/main/workshop-genai/financial-documents/csv-load/data/genai-data.nodes.__KGBuilder__.__Entity__.Executive.csv" AS row
MERGE (n:__index__ {`:ID`: row.`:ID`})
SET 
  n.chunk_index = toInteger(row.chunk_index),
  n.name = row.name,
  n:Executive;

LOAD CSV WITH HEADERS FROM "https://raw.githubusercontent.com/neo4j-graphacademy/workshop-genai/refs/heads/main/workshop-genai/financial-documents/csv-load/data/genai-data.nodes.__KGBuilder__.__Entity__.FinancialMetric.csv" AS row
MERGE (n:__index__ {`:ID`: row.`:ID`})
SET 
  n.chunk_index = toInteger(row.chunk_index),
  n.name = row.name,
  n:FinancialMetric;

LOAD CSV WITH HEADERS FROM "https://raw.githubusercontent.com/neo4j-graphacademy/workshop-genai/refs/heads/main/workshop-genai/financial-documents/csv-load/data/genai-data.nodes.__KGBuilder__.__Entity__.Product.csv" AS row
MERGE (n:__index__ {`:ID`: row.`:ID`})
SET 
  n.chunk_index = toInteger(row.chunk_index),
  n.name = row.name,
  n:Product;

LOAD CSV WITH HEADERS FROM "https://raw.githubusercontent.com/neo4j-graphacademy/workshop-genai/refs/heads/main/workshop-genai/financial-documents/csv-load/data/genai-data.nodes.__KGBuilder__.__Entity__.RiskFactor.csv" AS row
MERGE (n:__index__ {`:ID`: row.`:ID`})
SET 
  n.chunk_index = toInteger(row.chunk_index),
  n.name = row.name,
  n:RiskFactor;

LOAD CSV WITH HEADERS FROM "https://raw.githubusercontent.com/neo4j-graphacademy/workshop-genai/refs/heads/main/workshop-genai/financial-documents/csv-load/data/genai-data.nodes.__KGBuilder__.__Entity__.StockType.csv" AS row
MERGE (n:__index__ {`:ID`: row.`:ID`})
SET 
  n.chunk_index = toInteger(row.chunk_index),
  n.name = row.name,
  n:StockType;

LOAD CSV WITH HEADERS FROM "https://raw.githubusercontent.com/neo4j-graphacademy/workshop-genai/refs/heads/main/workshop-genai/financial-documents/csv-load/data/genai-data.nodes.__KGBuilder__.__Entity__.TimePeriod.csv" AS row
MERGE (n:__index__ {`:ID`: row.`:ID`})
SET 
  n.chunk_index = toInteger(row.chunk_index),
  n.name = row.name,
  n:TimePeriod;

LOAD CSV WITH HEADERS FROM "https://raw.githubusercontent.com/neo4j-graphacademy/workshop-genai/refs/heads/main/workshop-genai/financial-documents/csv-load/data/genai-data.nodes.__KGBuilder__.__Entity__.Transaction.csv" AS row
MERGE (n:__index__ {`:ID`: row.`:ID`})
SET 
  n.chunk_index = toInteger(row.chunk_index),
  n.name = row.name,
  n:Transaction;

LOAD CSV WITH HEADERS FROM "https://raw.githubusercontent.com/neo4j-graphacademy/workshop-genai/refs/heads/main/workshop-genai/financial-documents/csv-load/data/genai-data.nodes.__KGBuilder__.Chunk.csv" AS row
MERGE (n:__index__ {`:ID`: row.`:ID`})
SET 
  n.chunk_index = toInteger(row.index),
  n.text = row.text,
  n:Chunk
WITH n, row.embedding AS embedding
CALL db.create.setNodeVectorProperty(n, 'embedding', apoc.convert.fromJsonList(embedding));

LOAD CSV WITH HEADERS FROM "https://raw.githubusercontent.com/neo4j-graphacademy/workshop-genai/refs/heads/main/workshop-genai/financial-documents/csv-load/data/genai-data.nodes.__KGBuilder__.Company.__Entity__.csv" AS row
MERGE (n:__index__ {`:ID`: row.`:ID`})
SET 
  n.chunk_index = toInteger(row.chunk_index),
  n.name = row.name,
  n.ticker = row.ticker,
  n:Company;

LOAD CSV WITH HEADERS FROM "https://raw.githubusercontent.com/neo4j-graphacademy/workshop-genai/refs/heads/main/workshop-genai/financial-documents/csv-load/data/genai-data.nodes.__KGBuilder__.Document.csv" AS row
MERGE (n:__index__ {`:ID`: row.`:ID`})
SET 
  n.path = row.path,
  n.datetime = datetime(row.datetime),
  n:Document;

LOAD CSV WITH HEADERS FROM "https://raw.githubusercontent.com/neo4j-graphacademy/workshop-genai/refs/heads/main/workshop-genai/financial-documents/csv-load/data/genai-data.nodes.AssetManager.csv" AS row
MERGE (n:__index__ {`:ID`: row.`:ID`})
SET 
  n.managerName = row.managerName,
  n.managerCik = toInteger(row.managerCik),
  n:AssetManager;

LOAD CSV WITH HEADERS FROM "https://raw.githubusercontent.com/neo4j-graphacademy/workshop-genai/refs/heads/main/workshop-genai/financial-documents/csv-load/data/genai-data.nodes.Company.csv" AS row
MERGE (n:__index__ {`:ID`: row.`:ID`})
SET 
  n.name = row.name,
  n.ticker = row.ticker,
  n:Company;

LOAD CSV WITH HEADERS FROM "https://raw.githubusercontent.com/neo4j-graphacademy/workshop-genai/refs/heads/main/workshop-genai/financial-documents/csv-load/data/genai-data.nodes.Document.csv" AS row
MERGE (n:__index__ {`:ID`: row.`:ID`})
SET 
  n.path = row.path,
  n:Document;

LOAD CSV WITH HEADERS FROM "https://raw.githubusercontent.com/neo4j-graphacademy/workshop-genai/refs/heads/main/workshop-genai/financial-documents/csv-load/data/genai-data.relationships.FACES_RISK.csv" AS row
MATCH (a:__index__ {`:ID`: row.`:START_ID`})
MATCH (b:__index__ {`:ID`: row.`:END_ID`})
MERGE (a)-[:FACES_RISK]->(b);

LOAD CSV WITH HEADERS FROM "https://raw.githubusercontent.com/neo4j-graphacademy/workshop-genai/refs/heads/main/workshop-genai/financial-documents/csv-load/data/genai-data.relationships.FILED.csv" AS row
MATCH (a:__index__ {`:ID`: row.`:START_ID`})
MATCH (b:__index__ {`:ID`: row.`:END_ID`})
MERGE (a)-[:FILED]->(b);

LOAD CSV WITH HEADERS FROM "https://raw.githubusercontent.com/neo4j-graphacademy/workshop-genai/refs/heads/main/workshop-genai/financial-documents/csv-load/data/genai-data.relationships.FROM_CHUNK.csv" AS row
MATCH (a:__index__ {`:ID`: row.`:START_ID`})
MATCH (b:__index__ {`:ID`: row.`:END_ID`})
MERGE (a)-[:FROM_CHUNK]->(b);

LOAD CSV WITH HEADERS FROM "https://raw.githubusercontent.com/neo4j-graphacademy/workshop-genai/refs/heads/main/workshop-genai/financial-documents/csv-load/data/genai-data.relationships.FROM_DOCUMENT.csv" AS row
MATCH (a:__index__ {`:ID`: row.`:START_ID`})
MATCH (b:__index__ {`:ID`: row.`:END_ID`})
MERGE (a)-[:FROM_DOCUMENT]->(b);

LOAD CSV WITH HEADERS FROM "https://raw.githubusercontent.com/neo4j-graphacademy/workshop-genai/refs/heads/main/workshop-genai/financial-documents/csv-load/data/genai-data.relationships.HAS_METRIC.csv" AS row
MATCH (a:__index__ {`:ID`: row.`:START_ID`})
MATCH (b:__index__ {`:ID`: row.`:END_ID`})
MERGE (a)-[:HAS_METRIC]->(b);

LOAD CSV WITH HEADERS FROM "https://raw.githubusercontent.com/neo4j-graphacademy/workshop-genai/refs/heads/main/workshop-genai/financial-documents/csv-load/data/genai-data.relationships.ISSUED_STOCK.csv" AS row
MATCH (a:__index__ {`:ID`: row.`:START_ID`})
MATCH (b:__index__ {`:ID`: row.`:END_ID`})
MERGE (a)-[:ISSUED_STOCK]->(b);

LOAD CSV WITH HEADERS FROM "https://raw.githubusercontent.com/neo4j-graphacademy/workshop-genai/refs/heads/main/workshop-genai/financial-documents/csv-load/data/genai-data.relationships.MENTIONS.csv" AS row
MATCH (a:__index__ {`:ID`: row.`:START_ID`})
MATCH (b:__index__ {`:ID`: row.`:END_ID`})
MERGE (a)-[:MENTIONS]->(b);

LOAD CSV WITH HEADERS FROM "https://raw.githubusercontent.com/neo4j-graphacademy/workshop-genai/refs/heads/main/workshop-genai/financial-documents/csv-load/data/genai-data.relationships.NEXT_CHUNK.csv" AS row
MATCH (a:__index__ {`:ID`: row.`:START_ID`})
MATCH (b:__index__ {`:ID`: row.`:END_ID`})
MERGE (a)-[:NEXT_CHUNK]->(b);

LOAD CSV WITH HEADERS FROM "https://raw.githubusercontent.com/neo4j-graphacademy/workshop-genai/refs/heads/main/workshop-genai/financial-documents/csv-load/data/genai-data.relationships.OWNS.csv" AS row
MATCH (a:__index__ {`:ID`: row.`:START_ID`})
MATCH (b:__index__ {`:ID`: row.`:END_ID`})
MERGE (a)-[r:OWNS]->(b)
SET r.position_status = row.position_status,
    r.`Value` = toFloat(row.Value),
    r.shares = toInteger(row.shares),
    r.share_value = toFloat(row.share_value);

MATCH (n:__index__) REMOVE n:__index__;