# 📚 고등학교 데이터과학 RAG 챗봇

고등학교 2학년 데이터과학 교과서를 기반으로 질문에 답하는 **RAG(Retrieval-Augmented Generation) 기반 챗봇**입니다.  
LangChain + OpenAI + FAISS + Streamlit을 활용하여 구현되었습니다.

---

## 🚀 프로젝트 개요

이 프로젝트는 다음과 같은 흐름으로 동작합니다.

1. PDF 교과서를 로드  
2. 텍스트를 chunk 단위로 분할  
3. 임베딩 생성 후 FAISS 벡터 DB 저장  
4. 질문 입력 시 관련 문서 검색  
5. LLM이 문맥 기반 답변 생성  

👉 즉, **"교과서 기반 AI 튜터"** 역할을 수행합니다.

---

## 🧠 기술 스택

- LangChain
- OpenAI API
- FAISS
- Streamlit
- PyPDFLoader

---

## 📂 프로젝트 구조
📁 project/
│
├── data/
│ └── book.pdf
│
├── app.py
├── .env
└── README.md



---

## ⚙️ 설치 방법

### 1️⃣ 패키지 설치

```bash
pip install langchain langchain-community langchain-openai faiss-cpu streamlit python-dotenv
```

2️⃣ 환경 변수 설정 (.env)
OPENAI_API_KEY=your_api_key_here
▶️ 실행 방법
streamlit run app.py

브라우저에서 자동으로 실행됩니다.

💡 주요 코드 설명
📌 1. PDF 로드
loader = PyPDFLoader("data/고_데이터과학(김현철)_교과서.pdf")
pages = loader.load()


📌 2. 텍스트 분할
splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
docs = splitter.split_documents(pages)

📌 3. 벡터 DB 생성
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
vectordb = FAISS.from_documents(docs, embeddings)
retriever = vectordb.as_retriever(search_kwargs={"k": 3})

📌 4. RAG 체인 구성
rag_chain = (
    {"context": retriever, "question": RunnablePassthrough()}
    | prompt
    | ChatOpenAI(model="gpt-5-nano", temperature=0)
)

📌 5. Streamlit UI
import streamlit as st

st.header("고등학교 2학년 데이터과학 학습 챗봇")

query = st.text_input("무엇이 궁금한가요?")
submit = st.button("확인")

if submit:
    answer = rag_chain.invoke(query)
    st.success(f"질문: {query}\n\n답변: {answer.content}")


⚙️ 실행 방법 (한 번에 따라하기)
1️⃣ Conda 가상환경 생성
conda create -n py310 python=3.10

2️⃣ 가상환경 활성화
conda activate py310

3️⃣ 프로젝트 클론
git clone https://github.com/your-repo/rag-chatbot.git
cd rag-chatbot

4️⃣ 패키지 설치
pip install -r requirements.txt

5️⃣ 환경변수 설정 (.env)
OPENAI_API_KEY=your_api_key_here

6️⃣ 실행 (Streamlit)
streamlit run app.py

👉 브라우저에서 자동 실행: http://localhost:8501

📦 requirements.txt
langchain
langchain-community
langchain-openai
faiss-cpu
streamlit
python-dotenv
pypdf

🖥️ 실행 화면

<img src ="https://github.com/user-attachments/assets/a19a315a-af69-404b-84f6-0080970f2790">


👉 학생용 AI 튜터로 활용 가능

🎯 교육 활용 아이디어
✔ 수업 활용
데이터과학 개념 질의응답
수행평가 보조 도구
자기주도 학습 지원

✔ 프로젝트 확장
여러 교과서 통합 RAG
문제 생성 기능 추가
오답 분석 기능
음성 기반 AI 튜터

🔥 개선 아이디어 (고급)
Redis / Pinecone으로 확장
Retrieval 성능 튜닝 (k값, chunk size)
Prompt Engineering 고도화
Multi-Agent 구조 적용

⚠️ 주의사항
OpenAI API Key 필요
PDF 파일 경로 확인 필수
최초 실행 시 임베딩 생성 시간 소요

📌 향후 계획
학생 맞춤형 학습 추천
평가 자동 생성
학교 AI 플랫폼 연동

👨‍🏫 대상
고등학교 AI/데이터과학 수업
AI중점학교 프로젝트
교사 연수 실습 자료
📄 라이선스

MIT License
