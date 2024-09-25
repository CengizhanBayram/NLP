import faiss
from langchain_community.docstore.in_memory import InMemoryDocstore
from langchain_community.vectorstores import FAISS
import PyPDF2
from langchain import hub
from langchain_community.document_loaders import WebBaseLoader
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

# .env dosyasındaki ortam değişkenlerini yükle
load_dotenv()

# PDF'den metin çıkarma fonksiyonu
def extract_text_from_pdf(file_path):
    """
    Bu fonksiyon verilen PDF dosya yolundan metin çıkarır.
    """
    text = ""
    with open(file_path, 'rb') as file:
        reader = PyPDF2.PdfReader(file)
        for page in reader.pages:
            text += page.extract_text()
    return text

# PDF dosya yolunu belirtin
file_path = 'path_to_your_pdf_file.pdf'  # Bu kısmı kendi PDF dosya yolunuza göre değiştirin
pdf_text = extract_text_from_pdf(file_path)

# Metni belirli boyutlarda parçalara ayır
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=20)
documents = text_splitter.create_documents([pdf_text])

# Embedding modeli (OpenAI) tanımlama
embedding = OpenAIEmbeddings()

# FAISS vektör dizinini oluşturma
index = faiss.IndexFlatL2(embedding.embedding_dim)

# FAISS vektör mağazasını tanımlama
vector_store = FAISS(
    embedding_function=embedding,
    index=index,
    docstore=InMemoryDocstore(),
    index_to_docstore_id={},
)

# Belgeleri vektör mağazasına ekleme
vector_store.add_documents(documents)

# Belgeleri aramak için retriever tanımlama
retriever = vector_store.as_retriever()

# Belgeleri belirli bir formata dönüştürme işlevi
def format_docs(docs):
    """
    Belgeleri formatlar ve LLM'e uygun hale getirir.
    """
    return "\n\n".join([f"Document {i+1}:\n{doc.page_content}" for i, doc in enumerate(docs)])

# Prompt işlevi (Bağlam ve soruyu düzenleme)
def prompt(input_data):
    """
    Sorgu ve belgeleri birleştirerek LLM için bir giriş hazırlar.
    """
    context = input_data['context']
    question = input_data['question']
    return f"Context:\n{context}\n\nQuestion:\n{question}"

# LLM tanımlama
llm = ChatOpenAI(model_name="gpt-3.5-turbo")

# RAG zinciri (RAG Chain) tanımlama
rag_chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()} 
    | prompt 
    | llm 
    | StrOutputParser()
)

# Test sorgusu
test_query = "What is the main topic of the document?"

# RAG zincirini çalıştırma
result = rag_chain.invoke({"context": documents, "question": test_query})
print("Model Output:", result)
