from deepeval.models import DeepEvalBaseLLM
from langchain.schema.runnable import Runnable
import torch
from deepeval.models import DeepEvalBaseLLM
from langchain.schema.runnable import Runnable
from langchain.schema import AIMessage
import torch

class Deepeval(DeepEvalBaseLLM, Runnable):
    def __init__(self, model, name, tokenizer=None):
        self.model = model
        self.name = name
        self.tokenizer = tokenizer
        device = "cuda" if (torch.cuda.is_available()) else "cpu"
        self.device = device

    def load_model(self):
        return self.model
    
    def generate(self, prompt: str, **kwargs) -> str:
        model = self.load_model()
        response = model.invoke(prompt)
        
        # Ensure we return a plain string
        if isinstance(response, AIMessage):
            return response.content
        elif hasattr(response, 'content'):
            return response.content
        elif isinstance(response, dict) and 'content' in response:
            return response['content']
        else:
            return str(response)
    
    async def a_generate(self, prompt: str, **kwargs) -> str:
        return self.generate(prompt, **kwargs)

    def get_model_name(self):
        return self.name

    def invoke(self, input, config=None, **kwargs):
        if isinstance(input, dict) and "prompt" in input:
            prompt = input["prompt"]
        else:
            prompt = str(input)
        return self.generate(prompt, **kwargs)
