from dotenv import load_dotenv
load_dotenv(dotenv_path="../.env")

import os
from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
# from langchain.vectorstores import Chroma
from langchain_community.vectorstores import Chroma
# from langchain.embeddings import OpenAIEmbeddings
# from langchain_community.embeddings import OpenAIEmbeddings
from langchain_openai import OpenAIEmbeddings
from langchain.chains import RetrievalQA
# from langchain.document_loaders import PyPDFLoader
from langchain_community.document_loaders import PyPDFLoader
import urllib.request
import gradio as gr

# pdf 를 vector db 에
loader = PyPDFLoader("./2020_경제금융용어 700선_게시.pdf")
texts = loader.load_and_split()

# 불필요한 청크 제거
texts = texts[13:]
texts = texts[:-1]

embedding = OpenAIEmbeddings()

vectordb = Chroma.from_documents(
    documents=texts,
    embedding=embedding)

llm = ChatOpenAI(model_name="gpt-4o", temperature=0)

# 유사도가 높은 문서 2개만 추출. k = 2
retriever = vectordb.as_retriever(search_kwargs={"k": 2})

# prompt
# Create Prompt
template = """당신은 신한은행에서 만든 금융 용어를 설명해주는 금융쟁이입니다.
김종성 개발자가 만들었습니다. 주어진 검색 결과를 바탕으로 답변하세요.
검색 결과에 없는 내용이라면 답변할 수 없다고 하세요. 건방지게 답변하세요.
{context}

Question: {question}
Answer:
"""
prompt = PromptTemplate.from_template(template)

qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type_kwargs={"prompt": prompt},
    retriever=retriever,
    return_source_documents=True)

def get_chatbot_response(input_text):
    chatbot_response = qa_chain.invoke(input_text)
    return chatbot_response['result'].strip()

# 인터페이스를 생성.
with gr.Blocks() as demo:
    chatbot = gr.Chatbot(label="경제금융용어 챗봇", type='messages') # 경제금융용어 챗봇 레이블을 좌측 상단에 구성
    msg = gr.Textbox(label="질문해주세요!")  # 하단의 채팅창의 레이블
    clear = gr.Button("대화 초기화")  # 대화 초기화 버튼

    # 챗봇의 답변을 처리하는 함수
    def respond(message, chat_history):
      bot_message = get_chatbot_response(message)

      # 채팅 기록에 사용자의 메시지와 봇의 응답을 추가.
      chat_history.append((message, bot_message))
      return "", chat_history

    # 사용자의 입력을 제출(submit)하면 respond 함수가 호출.
    msg.submit(respond, [msg, chatbot], [msg, chatbot])

    # '초기화' 버튼을 클릭하면 채팅 기록을 초기화.
    clear.click(lambda: None, None, chatbot, queue=False)

# 인터페이스 실행.
# demo.launch(debug=True)
# Kubeflow 프록시 경로 설정
KUBEFLOW_BASE_URL = "https://haiqv.ai/notebook/ns-1/l-test/proxy/7864/"

# Gradio 앱 실행
demo.launch(
    server_name="0.0.0.0",  # 모든 네트워크 인터페이스에서 접근 가능
    server_port=7864,       # Kubeflow 프록시가 사용하는 포트
    share=False,
    inline=False,
    root_path=KUBEFLOW_BASE_URL
)