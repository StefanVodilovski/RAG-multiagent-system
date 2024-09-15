import os
import requests
import pandas as pd
import yaml
from dotenv import load_dotenv
from pathlib import Path
from llama_parse import LlamaParse
from llama_index.core.node_parser import MarkdownElementNodeParser
from llama_index.llms.huggingface import HuggingFaceInferenceAPI

# Load api keys
load_dotenv(Path(".env"))
HUGGINGFACE_API_KEY = os.getenv("HUGGINGFACE_API_KEY")
LLAMA_API_KEY = os.getenv("LLAMA_PARSER_KEY")

# Load project config
with open("config.yaml", "r") as file:
    config = yaml.safe_load(file)


def query(texts):
    api_url = f"https://api-inference.huggingface.co/pipeline/feature-extraction/{config['model_id']}"
    headers = {"Authorization": f"Bearer {HUGGINGFACE_API_KEY}"}
    response = requests.post(api_url,
                             headers=headers,
                             json={"inputs": texts,
                                   "options": {"wait_for_model": True}})
    return response.json()


def load_and_parse_documents():
    # Create parser
    parser = LlamaParse(
        api_key=LLAMA_API_KEY,  # you will need an API key, get it from https://cloud.llamaindex.ai/
        result_type="markdown"  # "markdown" and "text" are available
    )

    # Load and parse documents
    documents = [
        parser.load_data(f'google_data/google-{year}-environmental-report.pdf')
        for year in range(2021, 2024)
    ]

    llm = HuggingFaceInferenceAPI(
        model_name=config["hf_model_name"], token=HUGGINGFACE_API_KEY
    )

    node_parser = MarkdownElementNodeParser(llm=llm, num_workers=8)

    for document in documents:
        nodes = node_parser.get_nodes_from_documents(document)
        base_nodes, objects = node_parser.get_nodes_and_objects(nodes)

    return documents


def create_embeddings(documents):
    for year, document in zip(range(2021, 2024), documents):
        data = []
        for j, page in enumerate(document):
            embeddings = query(page.text)
            if embeddings:
                data.append({"id": j, "vector": embeddings, "text": page.text})

        df = pd.DataFrame(data)
        csv_file = f'google_data/google_embeddings/google_{year}.csv'
        df.to_csv(csv_file, index=False, encoding='utf-8')
        print(f"Embeddings saved to {csv_file}")


if __name__ == "__main__":
    parsed_documents = load_and_parse_documents()
    create_embeddings(parsed_documents)
