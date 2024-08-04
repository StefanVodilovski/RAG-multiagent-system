import pandas as pd
from pymilvus import (
    connections,
    Collection,
    FieldSchema,
    CollectionSchema,
    DataType,
    utility
)
import numpy as np


# Function to pad vectors
def pad_vector(vector, target_dim):
    padded_vector = np.zeros(target_dim)
    padded_vector[:len(vector)] = vector
    return padded_vector


def creating_schema():
    # Define the collection schema
    try:
        # Define the collection schema
        fields = [
            FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=False),
            FieldSchema(name="vector", dtype=DataType.FLOAT_VECTOR, dim=vector_dim),
            FieldSchema(name="content", dtype=DataType.VARCHAR, max_length=10000),
        ]
        schema = CollectionSchema(fields=fields, description="example collection")
        return schema
    except Exception as e:
        print(f"Error creating schema: {e}")
        raise


def creating_collection(schema):
    collection_name = "google_2022"

    # create the collection if it does not exist
    if not utility.has_collection(collection_name):
        collection = Collection(name=collection_name, schema=schema)
        print("Collection created successfully.")
    else:
        collection = Collection(name=collection_name)
        print("Collection already exists.")

    return collection


def prepare_entities():
    entities = []
    try:
        for index, row in df.iterrows():
            vector = eval(row['vector'])
            padded_vector = pad_vector(vector, vector_dim)
            entity = {
                "id": int(row['id']),
                "vector": padded_vector.tolist(),
                "content": row['text']
            }
            entities.append(entity)
        return entities
    except Exception as e:
        print(f"Error preparing entities: {e}")
        raise


def create_index(collection):
    # Define an index for the collection
    try:
        index_params = {
            "index_type": "HNSW",  # Hierarchical Navigable Small World
            "params": {"M": 16, "efConstruction": 200},  # Adjust parameters based on your use case
            "metric_type": "L2"  # Euclidean distance
        }
        collection.create_index(field_name="vector", index_params=index_params)
        print("Index created successfully.")
    except Exception as e:
        print(f"An error occurred while creating the index: {e}")


if __name__ == "__main__":
    # Read the CSV file and determine the vector dimension
    csv_file = "google_data/google_embeddings/google_2022.csv"
    df = pd.read_csv(csv_file)

    first_vector = df['vector'].apply(lambda x: eval(x)).iloc[0]
    vector_dim = len(first_vector)

    # Set up a Milvus client
    connections.connect(
        alias="default",
        uri="http://localhost:19530"
    )

    # Creating schema and collection
    schema = creating_schema()
    collection = creating_collection(schema)

    # Create an index if it doesn't exist
    if not collection.has_index():
        create_index(collection)

    # Prepare and insert data into Milvus
    mr = collection.insert(prepare_entities())
    print(f"Insert result: {mr}")
