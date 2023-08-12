__import__('pysqlite3')
import sys
import streamlit as st
import pandas as pd
#from pdf2image import convert_from_path
#import pytesseract
from PyPDF2 import PdfReader
from langchain.agents import create_csv_agent
from langchain.llms import OpenAI
import os
from langchain.document_loaders import TextLoader
from langchain.indexes import VectorstoreIndexCreator
import time
from langchain.document_loaders import PyPDFLoader
#import pypdf

sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

os.environ['OPENAI_API_KEY'] = st.secrets['OPENAI_API_KEY']

llm = OpenAI(temperature=0.1)

def load_csv_data(uploaded_file):
    df = pd.read_csv(uploaded_file)
    df.to_csv("uploaded_file.csv")
    return df


def load_txt_data(uploaded_file):
    with open('uploaded_file.txt', 'w') as f:
        f.write(uploaded_file.getvalue().decode())
    return uploaded_file.getvalue().decode()

def load_pdf_data(uploaded_file):
    with open('uploaded_file.txt', 'wb') as f:
        f.write(uploaded_file.getbuffer())
    pdf = PdfReader('uploaded_file.pdf')
    text = ""
    for page in pdf.pages:
        text += page.extract_text()
    #pdf_to_text('uploaded_file.pdf')
    return text

def main():
    st.title("Chat With Your Documents (csv, txt and pdf)")

    file = st.file_uploader("Upload a file", type=["csv", "txt", "pdf"])


    if file is not None:
        if file.type == "text/csv":
            doc = "csv"
            data = load_csv_data(file)
            agent = create_csv_agent(OpenAI(temperature=0), 'uploaded_file.csv', verbose=True)
            st.dataframe(data)

        elif file.type == "text/plain":
            doc = "text"
            data = load_txt_data(file)
            loader = TextLoader('uploaded_file.txt')
            index = VectorstoreIndexCreator().from_loaders([loader])

        elif file.type == "application/pdf":
            #doc = "text"
            #data = load_pdf_data(file)
            #loader = TextLoader('uploaded_file.pdf')
            loader = PyPDFLoader(file)
            index = VectorstoreIndexCreator().from_loaders([loader])

        # do something with the data


        question = st.text_input("Once uploaded, you can chat with your document. Enter your question here:")
        submit_button = st.button('Submit')

        if submit_button:
            if doc == "text":
                response = index.query(question)
            else:
                response = agent.run(question)

            if response:
                st.write(response)


if __name__ == "__main__":
    main()
