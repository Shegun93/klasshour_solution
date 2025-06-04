import os
import fitz
import re
from tqdm.auto import tqdm
from bertopic import BERTopic
from sentence_transformers import SentenceTransformer
import pandas as pd
import tiktoken
from sentence_transformers import SentenceTransformer
from langchain.text_splitter import RecursiveCharacterTextSplitter

pdf_folder = "/home/shegun93/Klasshour_Rags/Physics"
def text_formatter(text: str) -> str:
    cleaned = text.replace("\n"," ").strip()
    return cleaned

def parse_filename(filename: str) -> str:
    """
    Extracts the base name without extension from the filename.
    :param filename: The name of the file
    :return: Base name of the file
    """
    filename = os.path.basename(filename)[0]
    name = filename.split("_")
    return {
        "Subject": "Physics",
        "topic": name[0],
    }
