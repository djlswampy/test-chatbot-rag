from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS

load_dotenv()

# 1. 문서 로드
loader = PyPDFLoader("./data/박철수(가상인물).pdf")
docs = loader.load()

# 2. 청킹
splitter = CharacterTextSplitter(chunk_size=20, chunk_overlap=5)
chunks = splitter.split_documents(docs)

# 3. 임베딩 + 저장
embedding = OpenAIEmbeddings()
vectorstore = FAISS.from_documents(chunks, embedding)
vectorstore.save_local("faiss_index_박철수")