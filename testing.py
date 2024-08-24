# Import necessary libraries
import os
from pathlib import Path
from typing import Annotated

from dotenv import load_dotenv
from pymilvus import connections, Collection
import autogen
from autogen.agentchat import GroupChat, GroupChatManager
from autogen.agentchat.contrib.retrieve_user_proxy_agent import RetrieveUserProxyAgent
from PyPDF2 import PdfReader
import fitz  # Ensure you have PyMuPDF installed
from transformers import AutoTokenizer, AutoModelForCausalLM

# Set up the environment and embedding model
load_dotenv(Path(".env"))
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY
HUGGINGFACE_API_KEY = os.getenv("HUGGINGFACE_API_KEY")
os.environ["HUGGINGFACE_API_KEY"] = HUGGINGFACE_API_KEY


huggingface_config = {"model": "facebook/opt-350m"}
# Load Hugging Face model
hf_tokenizer = AutoTokenizer.from_pretrained(huggingface_config["model"])
hf_model = AutoModelForCausalLM.from_pretrained(huggingface_config["model"])

# Connect to Milvus
connections.connect(alias="default", host="localhost", port="19530")
collection_name = "google_2022"
collection = Collection(name=collection_name)

# Configuration for the language model
config_list = [
    {"model": "facebook/opt-350m"},
]

llm_config = {
    "config_list": config_list,
    "timeout": 60,
    "temperature": 0.8,
    "seed": 1234,
}

# Define termination message function
def termination_msg(x):
    return isinstance(x, dict) and "TERMINATE" == str(x.get("content", ""))[-9:].upper()

# Define the agents
boss = autogen.UserProxyAgent(
    name="Boss",
    is_termination_msg=termination_msg,
    human_input_mode="NEVER",
    code_execution_config=False,
    default_auto_reply="Reply `TERMINATE` if the task is done.",
    description="The boss who asks questions about the environmental report.",
)

boss_aid = RetrieveUserProxyAgent(
    name="Boss_Assistant",
    is_termination_msg=termination_msg,
    human_input_mode="NEVER",
    default_auto_reply="Reply `TERMINATE` if the task is done.",
    max_consecutive_auto_reply=3,
    retrieve_config={
        "task": "qa",
        "docs_path": "./google_data",
        "chunk_token_size": 10000,  # Handle large documents
        "model": "gpt_3.5",
        "client": "milvus",
        "collection_name": "google_2022",
        "get_or_create": True,
        "max_tokens": 2048,
        "must_break_at_empty_line": False,
    },
    code_execution_config=False,
    description="Assistant who answers questions based on the environmental report.",
)

google_env_expert = autogen.AssistantAgent(
    name="Google_Env_Expert",
    is_termination_msg=termination_msg,
    system_message="You are an expert at Google who has deep knowledge of environmental reports. Your role is to analyze the Google Environmental Report 2022 and provide insights and recommendations based on the report's content. Reply `TERMINATE` in the end when everything is done.",
    llm_config=llm_config,
    description="Google Environmental Expert who can analyze the environmental report and provide expert insights and recommendations."
)

# Define the group chat manager and agents
def _reset_agents():
    boss.reset()
    boss_aid.reset()

def generate_questions_and_answers():
    _reset_agents()
    def retrieve_content(
        message: Annotated[
            str,
            "Refined message which keeps the original meaning and can be used to retrieve content for code generation and question answering.",
        ],
        n_results: Annotated[int, "number of results"] = 3,
    ) -> str:
        boss_aid.n_results = n_results  # Set the number of results to be retrieved.
        _context = {"problem": message, "n_results": n_results}
        ret_msg = boss_aid.message_generator(boss_aid, None, _context)
        return ret_msg or message

    boss_aid.human_input_mode = "NEVER"  # Disable human input for boss_aid since it only retrieves content.

    for caller in [google_env_expert]:
        d_retrieve_content = caller.register_for_llm(
            description="retrieve content for answer generation and question answering.", api_style="function"
        )(retrieve_content)

    for executor in [boss]:
        executor.register_for_execution()(d_retrieve_content)

    groupchat = autogen.GroupChat(
        agents=[boss, google_env_expert],
        messages=[],
        max_round=3,
        speaker_selection_method="round_robin",
        allow_repeat_speaker=False,
    )

    manager = autogen.GroupChatManager(groupchat=groupchat, llm_config=llm_config)

    # Start chatting with the boss as this is the user proxy agent.
    boss.initiate_chat(
        manager,
        message="Analyze the Google Environmental Report 2022 and generate questions about its content. Provide insights and explanations based on the report.",
    )

# Run the process
generate_questions_and_answers()
