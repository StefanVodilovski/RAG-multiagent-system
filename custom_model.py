from types import SimpleNamespace

from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig

import autogen
from autogen import AssistantAgent, UserProxyAgent, ModelClient
from typing import Protocol, Optional, List, Union, Dict


# class ModelClient(Protocol):
#     """
#     A client class must implement the following methods:
#     - create must return a response object that implements the ModelClientResponseProtocol
#     - cost must return the cost of the response
#     - get_usage must return a dict with the following keys:
#         - prompt_tokens
#         - completion_tokens
#         - total_tokens
#         - cost
#         - model
#
#     This class is used to create a client that can be used by OpenAIWrapper.
#     The response returned from create must adhere to the ModelClientResponseProtocol but can be extended however needed.
#     The message_retrieval method must be implemented to return a list of str or a list of messages from the response.
#     """
#
#     RESPONSE_USAGE_KEYS = ["prompt_tokens", "completion_tokens", "total_tokens", "cost", "model"]
#
#     class ModelClientResponseProtocol(Protocol):
#         class Choice(Protocol):
#             class Message(Protocol):
#                 content: Optional[str]
#
#             message: Message
#
#         choices: List[Choice]
#         model: str
#
#     def create(self, params) -> ModelClientResponseProtocol:
#         ...
#
#     def message_retrieval(
#             self, response: ModelClientResponseProtocol
#     ) -> Union[List[str], List[ModelClient.ModelClientResponseProtocol.Choice.Message]]:
#         """
#         Retrieve and return a list of strings or a list of Choice.Message from the response.
#
#         NOTE: if a list of Choice.Message is returned, it currently needs to contain the fields of OpenAI's ChatCompletion Message object,
#         since that is expected for function or tool calling in the rest of the codebase at the moment, unless a custom agent is being used.
#         """
#         ...
#
#     def cost(self, response: ModelClientResponseProtocol) -> float:
#         ...
#
#     @staticmethod
#     def get_usage(response: ModelClientResponseProtocol) -> Dict:
#         """Return usage summary of the response using RESPONSE_USAGE_KEYS."""
#         ...

class CustomModelClient(ModelClient):
    def __init__(self, config, **kwargs):
        print(f"CustomModelClient config: {config}")
        self.device = config.get("device", "cpu")
        self.model = AutoModelForCausalLM.from_pretrained(config["model"]).to(self.device)
        self.model_name = config["model"]
        self.tokenizer = AutoTokenizer.from_pretrained(config["model"], use_fast=False)
        self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

        # params are set by the user and consumed by the user since they are providing a custom model
        # so anything can be done here
        gen_config_params = config.get("params", {})
        self.max_length = gen_config_params.get("max_length", 256)

        print(f"Loaded model {config['model']} to {self.device}")

    def create(self, params):
        if params.get("stream", False) and "messages" in params:
            raise NotImplementedError("Local models do not support streaming.")
        else:
            num_of_responses = params.get("n", 1)

            # can create my own data response class
            # here using SimpleNamespace for simplicity
            # as long as it adheres to the ClientResponseProtocol

            response = SimpleNamespace()

            inputs = self.tokenizer.apply_chat_template(
                params["messages"], return_tensors="pt", add_generation_prompt=True
            ).to(self.device)
            inputs_length = inputs.shape[-1]

            # add inputs_length to max_length
            max_length = self.max_length + inputs_length
            generation_config = GenerationConfig(
                max_length=max_length,
                eos_token_id=self.tokenizer.eos_token_id,
                pad_token_id=self.tokenizer.eos_token_id,
            )

            response.choices = []
            response.model = self.model_name

            for _ in range(num_of_responses):
                outputs = self.model.generate(inputs, generation_config=generation_config)
                # Decode only the newly generated text, excluding the prompt
                text = self.tokenizer.decode(outputs[0, inputs_length:])
                choice = SimpleNamespace()
                choice.message = SimpleNamespace()
                choice.message.content = text
                choice.message.function_call = None
                response.choices.append(choice)

            return response

    def message_retrieval(self, response):
        """Retrieve the messages from the response."""
        choices = response.choices
        return [choice.message.content for choice in choices]

    def cost(self, response) -> float:
        """Calculate the cost of the response."""
        response.cost = 0
        return 0

    @staticmethod
    def get_usage(response):
        # returns a dict of prompt_tokens, completion_tokens, total_tokens, cost, model
        # if usage needs to be tracked, else None
        return {}

