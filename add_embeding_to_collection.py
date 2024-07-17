from pymilvus import MilvusClient, connections, db
import requests
import nest_asyncio
from llama_parse import LlamaParse
from pymilvus import connections, db

from llama_index.core.node_parser import MarkdownElementNodeParser
from llama_index.llms.huggingface_api import HuggingFaceInferenceAPI


def add_embeding(pdf_files, collection_name):

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
    documents = parser.load_data(pdf_files)

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

    index = 120
    for i, document in enumerate(documents):

        emmbedings = query(document.text)
        data.append({"id": index, "vector": emmbedings, "text": document.text})
        index += 1

    conn = MilvusClient(
        uri="http://localhost:19530"
    )

    res = conn.list_collections()

    emmbedding_dim = len(data[1]["vector"])

    if collection_name not in res:

        conn.create_collection(
            collection_name=collection_name,
            dimension=emmbedding_dim,
            metric_type="IP",  # Inner product distance
            consistency_level="Strong",  # Strong consistency level
        )

        res = conn.get_load_state(
            collection_name=collection_name
        )

    conn.insert(collection_name=collection_name, data=data)


add_embeding(['google-2023-environmental-report.pdf'], 'google')
