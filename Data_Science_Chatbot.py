from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_core.runnables import RunnablePassthrough, RunnableParallel
from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv # .env 파일의 환경변수를 자동으로 불러오기 위한 모듈
load_dotenv()  # 실행 시 .env 파일을 찾아 변수들을 환경에 로드

# -----------------------------------------------------
# 1️⃣ PDF 로드
# -----------------------------------------------------
# Samsung_Card_Manual_Korean_1.3.pdf 파일을 읽어옵니다.
# PyPDFLoader는 PDF의 각 페이지를 text로 변환합니다.
loader = PyPDFLoader("data/book.pdf")
pages = loader.load()  # List[Document] 형태로 반환

# -----------------------------------------------------
# 2️⃣ 텍스트 분할 (chunk 단위)
# -----------------------------------------------------
# RecursiveCharacterTextSplitter로 문서를 청크 단위로 쪼갭니다.
# chunk_size와 overlap은 retrieval 성능에 영향을 줍니다.
splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
docs = splitter.split_documents(pages)


# -----------------------------------------------------
# 3️⃣ 임베딩 생성 및 벡터DB 저장
# -----------------------------------------------------
# OpenAI Embeddings를 이용해 각 문서 청크를 벡터화하고
# FAISS에 저장합니다. (로컬 DB로 빠른 검색 가능)
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
vectordb = FAISS.from_documents(docs, embeddings)

# 검색기(retriever) 객체 생성
retriever = vectordb.as_retriever(search_kwargs={"k": 3})  # 상위 3개 유사 문서 검색

# -----------------------------------------------------
# 4️⃣ 프롬프트 템플릿 정의
# -----------------------------------------------------
# 사용자의 질문과 검색된 문서를 결합해 LLM에 전달할 프롬프트를 구성합니다.
prompt = ChatPromptTemplate.from_template("""
너는 2학년 데이터과학 학습에 대한 전문 어시스턴트이다.
다음의 참고 문서를 바탕으로 질문에 정확하게 답하라.

[참고문서]
{context}

[질문]
{question}

한글로 간결하고 정확하게 답변하라.
""")


# LCEL(Runnable Sequence) 문법을 이용하여 pipeline을 정의
# Retriever로 문서를 검색하고, prompt로 결합한 뒤 LLM으로 생성하는 구조
rag_chain = (
    {"context": retriever, "question": RunnablePassthrough()}  # 입력을 question으로 받아 retriever 검색
    | prompt                                                    # prompt에 삽입
    | ChatOpenAI(model="gpt-5-nano", temperature=0)            # OpenAI LLM 호출
) 


# -----------------------------------------------------
# 6️⃣ 질의 수행
# -----------------------------------------------------
# 실제로 사용자가 질문을 던지는 부분입니다.
# query = "데이터 포인터트에 대해서 설명해줘"  # 예시 질의
# answer = rag_chain.invoke(query)

# # -----------------------------------------------------
# # 7️⃣ 결과 출력
# # -----------------------------------------------------
# print("질문:", query)
# print("답변:", answer.content)



import streamlit as st

st.header("고등학교 2학년 데이터과학 학습 챗봇")

query = st.text_input("무엇이 궁금한가요?")
# age = st.number_input("나이", min_value=0, max_value=120, value=25)
# lang = st.selectbox("언어 선택", ["Python", "R", "C++"])
submit = st.button("확인")

if submit:
    answer = rag_chain.invoke(query)
    st.success(f"질문은 {query}이며 , 답변은  {answer.content} 입니다.")