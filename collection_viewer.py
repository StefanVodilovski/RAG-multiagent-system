from pymilvus import MilvusClient, connections, db

client = MilvusClient(
    uri="http://localhost:19530"
)


res = client.describe_collection(
    collection_name="google"
)


print(res)
