from pymilvus import (
    connections,
    Collection,
    utility
)


def check_connection():
    if connections.has_connection("default"):
        print("Connection to Milvus is active.")
    else:
        print("Connection to Milvus is not active.")


def check_collection_data(collection_name):
    # Check if the collection exists and count the number of entities
    if utility.has_collection(collection_name):
        collection = Collection(name=collection_name)
        collection.load()
        entity_count = collection.num_entities
        print(f"Current count: {entity_count}.")
    else:
        print(f"Collection '{collection_name}' does not exist.")


def list_all_collections():
    collections = utility.list_collections()
    print(f"Existing collections: {collections}")


if __name__ == "__main__":
    # Set up a Milvus client
    connections.connect(
        alias="default",
        uri="http://localhost:19530"
    )

    # Check connection
    check_connection()

    # Check collection data
    collection_name = "google_2022"
    check_collection_data(collection_name)

    # List all existing collections
    list_all_collections()
