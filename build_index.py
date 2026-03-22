from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings

load_dotenv()

# -----------------------------------------------------
# 1. PDF 로드
# -----------------------------------------------------
loader = PyPDFLoader("data/book.pdf")
pages = loader.load()

# -----------------------------------------------------
# 2. 문서 분할
# -----------------------------------------------------
splitter = RecursiveCharacterTextSplitter(
    chunk_size=800,
    chunk_overlap=100
)
docs = splitter.split_documents(pages)

print(f"분할된 문서 수: {len(docs)}")

# -----------------------------------------------------
# 3. 임베딩 생성 및 FAISS 저장
# -----------------------------------------------------
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
vectordb = FAISS.from_documents(docs, embeddings)

# 로컬에 저장
vectordb.save_local("faiss_index")

print("FAISS 인덱스 저장 완료!")