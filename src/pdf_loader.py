from langchain_community.document_loaders import PyPDFLoader

def load_pdf(pdf_path: str):
    """
    Membaca dokumen PDF dan mengembalikannya
    dalam bentuk list of Document
    """
    loader = PyPDFLoader(pdf_path)
    documents = loader.load()
    return documents
