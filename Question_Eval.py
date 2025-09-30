from langchain.output_parsers import ResponseSchema, StructuredOutputParser
from langchain.prompts import ChatPromptTemplate, PromptTemplate
from langchain_ollama.llms import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate, HumanMessagePromptTemplate, SystemMessagePromptTemplate
import os
from datasets import load_dataset, Dataset
import pandas as pd
from Retrieval import Retrieval
from tqdm.auto import tqdm
from operator import itemgetter
from langchain_core.output_parsers import StrOutputParser
import torch
import random


Path="/home/shegun93/Klasshour_Rags/data.json"

question_schema = ResponseSchema(name="question", 
                                description="A question related to the context")
question_response_schema = [question_schema]

question_parser = StructuredOutputParser.from_response_schemas(question_response_schema)
format_instructions = question_parser.get_format_instructions()
bare_prompt_template = "{content}"
bare_template = ChatPromptTemplate.from_template(template=bare_prompt_template)

QA_template = """You are a professor in a university and you've been asked to create a test for students. for each context, create 
a question and make sure the question is specific to the context and avoid genertaing generic questions.

format response as JSON with the following structure:

question: the question based on the context

context: {context}
"""
prompt_template = ChatPromptTemplate.from_template(QA_template)

def load_data(File_path, split="train"):
    return load_dataset("json", data_files=File_path, split=split)


def Evaluation_Question_Generation(file_path, model, sample_size):
    llm = bare_template | model 
    dataset = load_data(file_path).shuffle(seed=42)
    question_list = []
    dataset_index = 0
    progress_bar = tqdm(total=sample_size, desc="Generating Question")

    while len(question_list) < sample_size:
        if dataset_index >= len(dataset):
            dataset = dataset.shuffle(seed=42)
            dataset_index = 0

        eval_data = dataset[dataset_index]
        dataset_index += 1

        messages = prompt_template.format_messages(
            context=eval_data,
            format_instructions=format_instructions
        )
        response = llm.invoke(messages)

        try:
            question = question_parser.parse(response)
            question["context"] = eval_data
            question_list.append(question)  # ✅ save result
            progress_bar.update(1)
            print(f"Generated Question: {question['question']}")
        except Exception as e:
            continue

    progress_bar.close()
    return question_list



