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
    # define the collection schema
    fields = [
        FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=False),
        FieldSchema(name="vector", dtype=DataType.FLOAT_VECTOR, dim=vector_dim),
        FieldSchema(name="content", dtype=DataType.VARCHAR, max_length=10000),
    ]

    schema = CollectionSchema(fields=fields, description="example collection")

    return schema


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


if __name__ == "__main__":
    # read the CSV file and determine the vector dimension
    csv_file = "google_data/google_embeddings/google_2022.csv"
    df = pd.read_csv(csv_file)

    first_vector = df['vector'].apply(lambda x: eval(x)).iloc[0]
    vector_dim = len(first_vector)

    # set up a Milvus client
    connections.connect(
        alias="default",
        uri="http://localhost:19530"
    )

    # creating schema and collection
    schema = creating_schema()
    collection = creating_collection(schema)

    # prepare and insert data into Milvus
    mr = collection.insert(prepare_entities())
    print(f"Insert result: {mr}")
