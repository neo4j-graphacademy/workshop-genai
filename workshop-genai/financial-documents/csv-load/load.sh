#!/bin/bash

# Load environment variables from .env file
if [ -f "../../../.env" ]; then
    export $(grep -v '^#' ../../../.env | xargs)
else
    echo "Error: .env file not found"
    exit 1
fi

# Check if required environment variables are set
if [ -z "$NEO4J_URI" ] || [ -z "$NEO4J_USERNAME" ] || [ -z "$NEO4J_PASSWORD" ]; then
    echo "Error: Required environment variables (NEO4J_URI, NEO4J_USERNAME, NEO4J_PASSWORD) are not set"
    exit 1
fi

# Execute the Cypher script using cypher-shell
echo "Loading data into Neo4j..."
cypher-shell -a "$NEO4J_URI" -u "$NEO4J_USERNAME" -p "$NEO4J_PASSWORD" -f load.cypher

if [ $? -eq 0 ]; then
    echo "Data loaded successfully!"
else
    echo "Error loading data"
    exit 1
fi

# Test the data load by counting nodes
echo ""
echo "Verifying data load..."
echo "Running node count query..."
cypher-shell -a "$NEO4J_URI" -u "$NEO4J_USERNAME" -p "$NEO4J_PASSWORD" "MATCH (n) RETURN count(n) AS totalNodes"
