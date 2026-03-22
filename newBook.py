from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings

load_dotenv()

# -----------------------------------------------------
# 1. 기존 FAISS 인덱스 로드
# -----------------------------------------------------
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

vectordb = FAISS.load_local(
    "faiss_index",
    embeddings,
    allow_dangerous_deserialization=True
)

print("기존 FAISS 인덱스 로드 완료!")

# -----------------------------------------------------
# 2. 새 PDF 로드
# -----------------------------------------------------
new_pdf = "data/book2.pdf"

loader = PyPDFLoader(new_pdf)
pages = loader.load()

print(f"새 문서 로드 완료: {len(pages)} 페이지")

# -----------------------------------------------------
# 3. 문서 분할
# -----------------------------------------------------
splitter = RecursiveCharacterTextSplitter(
    chunk_size=800,
    chunk_overlap=100
)

new_docs = splitter.split_documents(pages)

print(f"추가될 문서 수: {len(new_docs)}")

# -----------------------------------------------------
# 4. 기존 벡터DB에 추가
# -----------------------------------------------------
vectordb.add_documents(new_docs)

print("문서 추가 완료!")

# -----------------------------------------------------
# 5. 다시 저장
# -----------------------------------------------------
vectordb.save_local("faiss_index")

print("FAISS 인덱스 업데이트 완료!")