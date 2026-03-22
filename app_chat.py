from dotenv import load_dotenv
import streamlit as st

from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_core.runnables import RunnablePassthrough
from langchain_core.prompts import ChatPromptTemplate

load_dotenv()

st.set_page_config(
    page_title="고2 데이터과학 학습 챗봇",
    page_icon="📘",
    layout="centered"
)

st.header("📘 고2 데이터과학 학습 챗봇")
st.write("교재 PDF를 바탕으로 질문에 답하는 RAG 챗봇입니다.")

# -----------------------------------------------------
# 벡터DB 로드 (캐시)
# -----------------------------------------------------
@st.cache_resource
def load_retriever():
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

    vectordb = FAISS.load_local(
        "faiss_index",
        embeddings,
        allow_dangerous_deserialization=True
    )

    retriever = vectordb.as_retriever(search_kwargs={"k": 3})
    return retriever

# -----------------------------------------------------
# LLM 로드 (캐시)
# -----------------------------------------------------
@st.cache_resource
def load_llm():
    return ChatOpenAI(
        model="gpt-5-nano",
        temperature=0
    )

retriever = load_retriever()
llm = load_llm()

# -----------------------------------------------------
# 프롬프트 정의
# -----------------------------------------------------
prompt = ChatPromptTemplate.from_template("""
당신은 고등학교 2학년 데이터과학 학습을 돕는 전문 도우미입니다.
다음 참고 문서를 바탕으로 질문에 정확하게 답하세요.

[참고 문서]
{context}

[질문]
{question}

답변은 한국어로 간결하고 정확하게 작성하세요.
""")

# -----------------------------------------------------
# RAG 체인 구성
# -----------------------------------------------------
rag_chain = (
    {"context": retriever, "question": RunnablePassthrough()}
    | prompt
    | llm
)

# -----------------------------------------------------
# 사용자 입력
# -----------------------------------------------------
query = st.text_input("무엇이 궁금한가요?")

submit = st.button("질문하기")

if submit:
    if not query.strip():
        st.warning("질문을 입력해주세요.")
    else:
        with st.spinner("답변 생성 중입니다..."):
            answer = rag_chain.invoke(query)

        st.success("답변 생성 완료!")
        st.write("### 답변")
        st.write(answer.content)