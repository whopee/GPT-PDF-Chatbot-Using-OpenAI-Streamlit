import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.llms import OpenAI
from langchain.chains.question_answering import load_qa_chain
import pickle

st.header('ChatPDF Using Streamlit and OpenAI')

pdf = st.file_uploader('Upload a PDF file with text in English. PDFs that only contain images will not be recognized.', type=['pdf']) 
query = st.text_input('Ask a question about the PDF you entered!', max_chars=300)

if pdf is not None:
    try:
        pdf_doc = PdfReader(pdf)
        txt = ""
        for page in pdf_doc.pages:
            txt += page.extract_text()
        
        text_split = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len
        )
        chunks = text_split.split_text(text=txt)

        embeddings = OpenAIEmbeddings()
        vectorStore = FAISS.from_texts(chunks, embedding=embeddings)

        docs = vectorStore.similarity_search(query=query)
        llm = OpenAI(model_name="gpt-3.5-turbo")

        chain = load_qa_chain(llm=llm, chain_type="stuff")

        response = chain.run(input_documents=docs, question=query)
        st.write(response)

        store_name = "STORE_NAME"  # define the store name
        with open(f"{store_name}.pkl", "wb") as f:
            pickle.dump(vectorStore, f)

    except Exception as e:
        st.error("An error occurred: " + str(e))
else:
    st.warning("Please upload a PDF file.")
