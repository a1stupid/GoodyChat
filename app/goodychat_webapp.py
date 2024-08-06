from langchain.document_loaders import TextLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings.huggingface import HuggingFaceEmbeddings
import os
from langchain.vectorstores import Chroma
from langchain_openai import ChatOpenAI
from langchain.chains import RetrievalQA

def response_generator(prompt):
    add_new_datasets = False
    if add_new_datasets:
        # 导入文本
        filepath = "../knowledges/rumor.txt"
        loader = TextLoader(filepath, autodetect_encoding=True)
        # 切分文本
        textsplitter = CharacterTextSplitter()
        split_docs = loader.load_and_split(textsplitter)
        print(split_docs)

    # 初始化embeddings
    embeddings = HuggingFaceEmbeddings(
        model_name="/home/lzy22/my_projects/goody_chat/models/m3e-base",
        model_kwargs={'device': "cuda:0"},
        encode_kwargs={'normalize_embeddings': False}
    )
    #使用chroma存储数据
    if not os.path.exists("../VectorStore/text_VectorStore"):
        db = Chroma.from_documents(split_docs, embeddings, persist_directory="../VectorStore/text_VectorStore")
    else:
        db = Chroma(persist_directory="../VectorStore/text_VectorStore", embedding_function=embeddings)

    # 创建问答对象
    retriever = db.as_retriever()
    llm = ChatOpenAI(
        model_name="Qwen-1_8B-Chat",
        openai_api_base="http://127.0.0.1:6000/v1",
        openai_api_key="EMPTY",
        temperature=0,
    )
    qa = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=retriever, return_source_documents=True)
    # 进行问答
    result = qa(prompt)
    return result['result']


#使用streamlit构建聊天机器人 streamlit run goodychat_webapp.py
import streamlit as st

# 初始化聊天记录
if "messages" not in st.session_state:
    st.session_state.messages = []
# 展示聊天记录
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# 对用户的输入做出反应
if prompt := st.chat_input("给GoodyChat发送消息"):
    # 在聊天消息容器中显示用户消息
    with st.chat_message("user"):
        st.markdown(prompt)
    # 将用户消息添加到聊天记录
    st.session_state.messages.append({"role": "user", "content": prompt})

    # 在聊天消息容器中显示assistant响应
    with st.chat_message("assistant"):
        response = st.write_stream([response_generator(prompt)])
    # 添加assistant回复聊天记录
    st.session_state.messages.append({"role": "assistant", "content": response})
