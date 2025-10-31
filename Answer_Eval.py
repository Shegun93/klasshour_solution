from datasets import load_dataset
import pandas as pd
from Question_Eval import Evaluation_Question_Generation
from langchain_ollama import OllamaLLM
from langchain.prompts import ChatPromptTemplate
from langchain.output_parsers import ResponseSchema, StructuredOutputParser
from tqdm.auto import tqdm
import os
Answer_schema = ResponseSchema(name="answer", 
                                description="An answer to the context")
answer_response_schema = [Answer_schema]

Answer_parser = StructuredOutputParser.from_response_schemas(answer_response_schema)
bare_prompt_template = "{content}"
format_instructions = Answer_parser.get_format_instructions()
bare_template = ChatPromptTemplate.from_template(template=bare_prompt_template)
Answer_template = """
You are a professor in university and you've been asked to create a test for students. for each  question and context, create 
an question

format response as JSON with the following keys:

answer

question: {question}
context: {context}
"""
prompt_template = ChatPromptTemplate.from_template(template=Answer_template)

def Answers_Questions(question_list, model, save_folder):
    llm = bare_template | model
    answers = []

    for qa in tqdm(question_list, desc="Answering Questions"):
        question = qa["question"]
        context = qa["context"]

        answer_template = ChatPromptTemplate.from_template(
            """You are a knowledgeable assistant. 
            Given the context below, provide a clear and concise answer 
            to the following question.

            Context: {context}
            Question: {question}
            """
        )

        messages = answer_template.format_messages(
            context=context,
            question=question
        )
        response = llm.invoke(messages)

        answers.append({
            "context": context,
            "question": question,
            "answer": response
        })

    df = pd.DataFrame(answers)
    os.makedirs(save_folder, exist_ok=True)
    csv_path = os.path.join(save_folder, "qa_results.csv")
    json_path = os.path.join(save_folder, "qa_results.json")
    df.to_csv(csv_path, index=False)
    df.to_json(json_path, orient="records", indent=2)

    return df

if __name__ == "__main__":
    Path = "/home/shegun93/Klasshour_Rags/data.json"
    Model = OllamaLLM(model="llama2:7b", temperature=0.7) 

    questions = Evaluation_Question_Generation(
        file_path=Path,
        model=Model,
        sample_size=5
    )


    answered_df = Answers_Questions(questions, Model, save_folder="results")

    print(answered_df.head())