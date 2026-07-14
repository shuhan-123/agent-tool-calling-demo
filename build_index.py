"""
运行一次即可：把论文 PDF 切片、向量化，存成本地 FAISS 索引。
用法：python build_index.py 你的论文.pdf
"""

import sys
from pathlib import Path
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings

BASE_DIR = Path(__file__).parent
INDEX_DIR = BASE_DIR / "faiss_index"


def main():
    if len(sys.argv) < 2:
        print("用法：python build_index.py 论文路径.pdf")
        sys.exit(1)

    pdf_path = sys.argv[1]
    print(f"正在读取 {pdf_path} ...")
    loader = PyPDFLoader(pdf_path)
    pages = loader.load()

    print("正在切片...")
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = splitter.split_documents(pages)
    print(f"共切成 {len(chunks)} 个片段")

    print("正在向量化并建索引（首次运行会自动下载模型，需要几分钟）...")
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
    vectorstore = FAISS.from_documents(chunks, embeddings)
    vectorstore.save_local(str(INDEX_DIR))
    print(f"索引已保存到 {INDEX_DIR}")


if __name__ == "__main__":
    main()
