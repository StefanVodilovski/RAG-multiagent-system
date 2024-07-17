from pymilvus import MilvusClient, connections, db
import requests
import nest_asyncio
from llama_parse import LlamaParse
from pymilvus import connections, db

from llama_index.core.node_parser import MarkdownElementNodeParser
from llama_index.llms.huggingface_api import HuggingFaceInferenceAPI


def query(texts):
    response = requests.post(api_url, headers=headers, json={
                             "inputs": texts, "options": {"wait_for_model": True}})
    return response.json()


# database = db.create_database("book")
parser = LlamaParse(
    # you will need an API key, get it from https://cloud.llamaindex.ai/
    api_key="llx-16bjSs4EzcUSA1uQ5wd9PW5SFt9hX41yb1W5DdlroZN6XOBB",
    result_type="markdown"  # "markdown" and "text" are available
)

nest_asyncio.apply()
documents = parser.load_data("google-2024-environmental-report.pdf")


llm = HuggingFaceInferenceAPI(
    model_name="HuggingFaceH4/zephyr-7b-alpha", token="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"
)

node_parser = MarkdownElementNodeParser(
    llm=llm, num_workers=8)  # .from_defaults()

# Retrieve nodes (text) and objects (table)
nodes = node_parser.get_nodes_from_documents(documents)

base_nodes, objects = node_parser.get_nodes_and_objects(nodes)

model_id = "sentence-transformers/all-MiniLM-L6-v2"
hf_token = "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"


api_url = f"https://api-inference.huggingface.co/pipeline/feature-extraction/{model_id}"
headers = {"Authorization": f"Bearer {hf_token}"}


for document in documents:
    print(document.text)
    print(document)
    break


data = []

for i, document in enumerate(documents):

    emmbedings = query(document.text)
    data.append({"id": i, "vector": emmbedings, "text": document.text})

conn = MilvusClient(
    uri="http://localhost:19530"
)

res = conn.list_collections()

print(res)


collection_name = "google"


emmbedding_dim = len(data[1]["vector"])


conn.create_collection(
    collection_name=collection_name,
    dimension=emmbedding_dim,
    metric_type="IP",  # Inner product distance
    consistency_level="Strong",  # Strong consistency level

)

res = conn.get_load_state(
    collection_name=collection_name
)

print(res)


conn.insert(collection_name=collection_name, data=data)
