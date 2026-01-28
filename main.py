from src.pdf_loader import load_pdf
from src.text_splitter import split_documents
from src.vector_store import create_vector_store
from src.rag_chain import answer_question

PDF_PATH = "data/pedoman_akademik.pdf"

def main():
    print("Membaca PDF...")
    documents = load_pdf(PDF_PATH)

    print("Chunking dokumen...")
    chunks = split_documents(documents)

    print("Membuat vector store (Chroma)...")
    vector_store = create_vector_store(chunks)

    print("AI Asisten Akademik (LM Studio) Siap Digunakan!\n")

    while True:
        question = input("Pertanyaan (ketik 'exit'): ")
        if question.lower() == "exit":
            print("Keluar dari sistem.")
            break

        answer, sources = answer_question(vector_store, question)

        print("Jawaban:")
        print(answer)

        print("Referensi Dokumen:")
        for doc in sources:
            page = doc.metadata.get("page", "Tidak diketahui")
            print(f"- Halaman {page}")

        print("-" * 60)

if __name__ == "__main__":
    main()
