import streamlit as st
from PyPDF2 import PdfReader
from langchain_text_splitter import RecursiveCharacterTextSplitter      
import os

from langchain_google_genai import GoogleGenerativeAIEmbeddings
import google.generativeai as genai
from langchain_community.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate

load_dotenv()

genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

def get_pdf_text(pdf_docs):
    text=""
    for pdf in pdf_docs:
        pdf_reader=PdfReader(pdf)
        for page in pdf_reader.pages:
            text+=page.extract_text()   
    return text

def get_text_chunks(text):
    text_splitter=RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
    chunks=text_splitter.split_text(text)
    return chunks

def get_vector_store(text_chunks):
    embeddings=GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vector_store=FAISS.from_texts(texts=text_chunks, embedding=embeddings)
    vector_store.save_local("faiss_index")


def get_conversation_chain(vector_store):
    prompt_template="""
    You are a helpful assistant answering questions about the uploaded PDF documents.
    Use the following pieces of context to answer the question.
    If you don't know the answer, just say that you don't know, don't try to make up an answer.

    Context: {context}
    Question: {question}

    Answer:
    """
    model=ChatGoogleGenerativeAI(model="gemini-pro", temperature=0.3)

    prompt=PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    chain=load_qa_chain(model, chain_type="stuff", prompt=prompt)
    return chain

def user_input(user_question):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")

    new_db = FAISS.load_local("faiss_index", embeddings)
    docs = new_db.similarity_search(user_question)

    chain = get_conversation_chain(new_db)

    response = chain(
        {"input_documents": docs, "question": user_question}
        ,return_only_outputs=True)
    
    print(response)
    st.write("Reply:",response["output_text"])

def main():
    st.set_page_config("Chat with Mutiple pdf")
    st.header("Chat with Mutiple PDF Documents")

    user_question=st.text_input("Ask a question about your PDF documents:  ")

    if user_question:
        user_input(user_question)

    with st.sidebar():
        st.title("Menu:")
        pdf_docs=st.file_uploader("Upload your PDF documents here and click on 'Process'", accept_multiple_files=True)
        if st.button("Submit & Process"):
            with st.spinner("Processing..."):
                # Get the PDF text
                raw_text=get_pdf_text(pdf_docs)

                # Get the text chunks
                text_chunks=get_text_chunks(raw_text)

                # Create the vector store
                get_vector_store(text_chunks)
                st.success("Done")
                

if __name__=="__main__":
    main()
