from langchain_community.vectorstores.faiss import FAISS
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAI, OpenAIEmbeddings, ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.chains.question_answering import load_qa_chain
from langchain.docstore.document import Document
from dotenv import load_dotenv
from getpass import getpass
import streamlit as st
import os
import time
 
load_dotenv()  
os.getenv("OPENAI_API_KEY")
 
def get_log_file(log_doc):
    doc = TextLoader(log_doc, encoding = 'UTF-8')
    docs = doc.load()
    return docs
 
def doc_splitter(docs):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000,chunk_overlap=15)
    splitted_docs = text_splitter.split_documents(docs)
    return splitted_docs
 
def get_vector_store(splitted_docs):
    embeddings = OpenAIEmbeddings(model = "text-embedding-3-small")
    vector_store = FAISS.from_documents(splitted_docs, embedding=embeddings)
    vector_store.save_local("faiss_index")
 
def get_conversational_chain():
 
    prompt_template = """
    Answer the question as detailed as possible from the provided context, make sure to provide all the details, if the answer is not in
    provided context just say, "answer is not available in the context", don't provide the wrong answer\n\n or answers out of the context
    Context:\n {context}?\n
    Question: \n{question}\n
 
    Answer:
    """
 
    model = OpenAI(model_name="gpt-3.5-turbo-instruct", temperature=0.3)
    prompt = PromptTemplate(template = prompt_template, input_variables = ["context", "question"])
    chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)
 
    return chain
 
def summary_docs(docs:list):
    summary = []
    number_of_docs = 80
    prompt = '''
    system: You are a log summarizer. You will analyze the log entries in given context return the summarized data of providex context with number of occurences make it precise and accurate.
    context: {context}\n
    Answer:
    '''
    model = OpenAI(model_name="gpt-3.5-turbo-instruct", temperature=0.3)
    prompt = PromptTemplate(template=prompt, input_variables = ["context"])
    chain = load_qa_chain(model, chain_type="stuff", prompt = prompt)
    while number_of_docs != 0:
        doc = docs[0:10]
        docs = docs[10:]
        temp = chain({"input_documents": doc} , return_only_outputs=True)
        summary.append(temp["output_text"])
        number_of_docs -= 10
    return summary
 
 
def search_similarity(user_input):
    embeddings = OpenAIEmbeddings(model = "text-embedding-3-small")
    new_db = FAISS.load_local("faiss_index", embeddings,allow_dangerous_deserialization=True)
    docs = new_db.similarity_search(user_input, k= 80, fetch_k= 80)
    source = docs[0].metadata
    summary = summary_docs(docs=docs)
    summary = str.join(' ',summary)
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000,chunk_overlap=15)
    document = Document(page_content=summary)
    document.metadata = {"source":source}
    document = [document]
    splitted_docs = text_splitter.split_documents(document)
    chain = get_conversational_chain()
    response = chain(
        {"input_documents":splitted_docs, "question": user_input}
        , return_only_outputs=True)
    st.write("Reply: ", response["output_text"])
 
def main():
    st.set_page_config(page_title="Log Analyzer")
    st.header("Chat with the bot to get insights from the Log Files")
 
    user_input = st.text_input("Ask a Question from the Log File")
 
    if user_input:
        search_similarity(user_input=user_input)
 
    with st.sidebar:
        st.title("Menu:")
        doc_names = []
        log_doc = st.file_uploader("Upload your Log Files and Click on the Submit & Process Button", type=["log"], accept_multiple_files=True)
        if st.button("Submit & Process"):
            with st.spinner("Processing..."):
                for doc in log_doc:
                    doc_names.append(doc.name)
                    with open(os.path.join("./log_files", doc.name), "wb") as f:
                        f.write(doc.getbuffer())
                directory = os.listdir("./log_files")
                for doc in directory:
                    if doc in doc_names:
                        docs = get_log_file(log_doc=os.path.join("./log_files", doc))
                        text_chunks = doc_splitter(docs=docs)
                        get_vector_store(splitted_docs=text_chunks)
                        st.success("Done")
 
 
if __name__ == "__main__":
    main()