import os
import pdfplumber
import streamlit as st
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from tenacity import Retrying, stop_after_attempt, wait_exponential, retry



def extract_text_from_pdf(file):
    with pdfplumber.open(file) as pdf:
        pages = [page.extract_text() for page in pdf.pages]
    return "\n".join(pages)

def split_text_in_to_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=20,
        length_function=len,
        separators=["\n\n", "\n", " ", ""]
    )
    split_text = text_splitter.split_text(text)
    print ('split_text',split_text)
    return split_text

def perform_embedding_on_chunks(split_text):
    embeddings = OpenAIEmbeddings(openai_api_key=os.getenv("OPENAI_API_KEY"))
    embedding_index = FAISS.from_texts(split_text, embeddings)
    return embedding_index 

def find_similar_texts(embedding_index, question):
    similar_text = embedding_index.similarity_search(question)
    return similar_text

def get_response_from_gpt(text, question):
    response = openai.Completion.create(
      engine="text-davinci-002",
      prompt=f"{text}\n\nQuestion: {question}\nAnswer:",
      temperature=0,
      max_tokens=150,
      top_p=1.0,
      frequency_penalty=0.0,
      presence_penalty=0.0
    )
    return response.choices[0].text.strip()

@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=1, max=10))
def embeddings_client_create(**kwargs):
    return embeddings.client.create(**kwargs)

def get_embeddings(text, engine):
    response = embeddings_client_create(input=text, model=engine)
    return response['data'][0]['embedding']

# Streamlit app
st.title("PDF Chatbot")

uploaded_file = st.file_uploader("Choose a PDF file", type="pdf")
if uploaded_file is not None:
    text = extract_text_from_pdf(uploaded_file)
    
    split_text = split_text_in_to_chunks(text)
    st.write("PDF text split in to chunks.")
    embedding_index = perform_embedding_on_chunks(split_text)
    st.write("embedding peformed on chunks.")
    st.write("PDF text extracted. You can now ask questions.")

    user_question = st.text_input("Ask a question:")
    if user_question:
        try:
            similar_text = find_similar_texts(embedding_index, user_question)
            if similar_text:
                gpt_response = get_response_from_gpt(similar_text, user_question)
                st.write(gpt_response)
            else:
                st.write("Sorry, we couldn't find any similar texts.")
        except RateLimitError as e:
            st.write(f"Rate limit exceeded. Please wait and try again. Error: {e}")
