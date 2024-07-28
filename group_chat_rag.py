import autogen
from autogen import *
from autogen.agentchat.contrib.retrieve_user_proxy_agent import RetrieveUserProxyAgent
from pymilvus import MilvusClient
from sqlalchemy.sql.annotation import Annotated

config_list = autogen.config_list_from_json("models_config")

print("LLM models: ", [config_list[i]["model"] for i in range(len(config_list))])


# Database Connection
client = MilvusClient(uri="http://localhost:19530")

# 1. Constructing Agents
def termination_msg(x):
    return isinstance(x, dict) and "TERMINATE" == str(x.get("content", ""))[-9:].upper()


llm_config = {"model": "core42/jais-30b-v1", "timeout": 60, "temperature": 0.8, "seed": 1234}

boss = autogen.UserProxyAgent(
    name="Boss",
    is_termination_msg=termination_msg,
    human_input_mode="NEVER",
    code_execution_config=False,  # we don't want to execute code in this case.
    default_auto_reply="Reply `TERMINATE` if the task is done.",
    description="The boss who ask questions and give tasks.",
)

environmentalist_aid = RetrieveUserProxyAgent(
    name="En",
    is_termination_msg=termination_msg,
    human_input_mode="NEVER",
    default_auto_reply="Reply `TERMINATE` if the task is done.",
    max_consecutive_auto_reply=3,
    retrieve_config={
        "task": "code",
        "model": config_list[0]["model"],
        "collection_name": "google_test_4",
        "get_or_create": True,
        "milvus_client": client,  # Connect to Milvus server
    },
    code_execution_config=False,  # we don't want to execute code in this case.
    description="Assistant who has extra content retrieval power for solving difficult problems.",
)

environmentalist_officer = AssistantAgent(
    name="Environmental Compliance Officer",
    is_termination_msg=termination_msg,
    llm_config=None,
    system_message="You oversee environmental policies, compliance with regulations, and the preparation of environmental reports that detail the company's impact on the environment. Reply `TERMINATE` in the end when everything is done.",
    description="Environmental Compliance Officer who answers to any question related to compliance with regulations, and the preparation of the environmental reports that detail the company's impact on the environment.",
)


PROBLEM = "How was your environmental compliance in the year 2021."

def _reset_agents():
    boss.reset()
    environmentalist_aid.reset()
    environmentalist_officer.reset()


def rag_chat():
    _reset_agents()
    groupchat = autogen.GroupChat(
        agents=[environmentalist_aid, environmentalist_officer], messages=[], max_round=12, speaker_selection_method="round_robin"
    )
    manager = autogen.GroupChatManager(groupchat=groupchat, llm_config=llm_config)

    # Start chatting with boss_aid as this is the user proxy agent.
    environmentalist_aid.initiate_chat(
        manager,
        message=environmentalist_aid.message_generator,
        problem=PROBLEM,
        n_results=3,
    )

def call_rag_chat():
    _reset_agents()

    # In this case, we will have multiple user proxy agents and we don't initiate the chat
    # with RAG user proxy agent.
    # In order to use RAG user proxy agent, we need to wrap RAG agents in a function and call
    # it from other agents.
    def retrieve_content(
        message: Annotated[
            str,
            "Refined message which keeps the original meaning and can be used to retrieve content for code generation and question answering.",
        ],
        n_results: Annotated[int, "number of results"] = 3,
    ) -> str:
        environmentalist_aid.n_results = n_results  # Set the number of results to be retrieved.
        # Check if we need to update the context.
        update_context_case1, update_context_case2 = environmentalist_aid._check_update_context(message)
        if (update_context_case1 or update_context_case2) and environmentalist_aid.update_context:
            environmentalist_aid.problem = message if not hasattr(environmentalist_aid, "problem") else environmentalist_aid.problem
            _, ret_msg = environmentalist_aid._generate_retrieve_user_reply(message)
        else:
            _context = {"problem": message, "n_results": n_results}
            ret_msg = environmentalist_aid.message_generator(environmentalist_aid, None, _context)
        return ret_msg if ret_msg else message

    environmentalist_aid.human_input_mode = "NEVER"  # Disable human input for boss_aid since it only retrieves content.

    for caller in [environmentalist_officer]:
        d_retrieve_content = caller.register_for_llm(
            description="retrieve content for code generation and question answering.", api_style="function"
        )(retrieve_content)

    for executor in [boss]:
        executor.register_for_execution()(d_retrieve_content)

    groupchat = autogen.GroupChat(
        agents=[boss, environmentalist_officer],
        messages=[],
        max_round=12,
        speaker_selection_method="round_robin",
        allow_repeat_speaker=False,
    )

    manager = autogen.GroupChatManager(groupchat=groupchat, llm_config=llm_config)

    # Start chatting with the boss as this is the user proxy agent.
    boss.initiate_chat(
        manager,
        message=PROBLEM,
    )


rag_chat()
