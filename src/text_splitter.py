from langchain_text_splitters import RecursiveCharacterTextSplitter

def split_documents(documents):
    """
    Memecah dokumen menjadi chunk kecil agar optimal untuk embedding
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=100
    )
    return splitter.split_documents(documents)
