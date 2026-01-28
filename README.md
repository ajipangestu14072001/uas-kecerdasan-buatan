# AI Asisten Akademik Berbasis RAG (PDF)

## Deskripsi
Sistem ini adalah implementasi Retrieval-Augmented Generation (RAG) yang dirancang untuk menjawab pertanyaan seputar regulasi kampus berdasarkan dokumen PDF Pedoman Akademik. Sistem ini bekerja secara fully local menggunakan LM Studio untuk menjaga privasi data dan efisiensi biaya.

## Arsitektur
1. Ingestion: Mengekstraksi teks dari PDF menggunakan PyPDFLoader.
2. Smart Chunking: Memecah dokumen menjadi potongan kecil (RecursiveCharacterTextSplitter) dengan ukuran 500 karakter dan overlap 100 karakter untuk menjaga konteks.
3. Vector Store: Mengubah teks menjadi vektor menggunakan HuggingFaceEmbeddings (model: all-MiniLM-L6-v2) dan menyimpannya ke database Chroma.
4. Retrieval: Mengambil top-K dokumen relevan dan melakukan filtrasi konten unik untuk mencegah duplikasi.
5. Generation: Memproses konteks melalui Gemma 3 via API lokal untuk menghasilkan jawaban natural.

## Teknologi
- Python
- Orchestration: LangChain 0.2
- Vector Database: ChromaDB
- HuggingFace Embeddings
- LLM Engine: LM Studio (Local Server)
- Model: Google Gemma-3-4B-IT

## Cara Menjalankan
1. Aktifkan virtual environment
2. Install dependency
   pip install -r requirements.txt
3. Jalankan aplikasi
   python main.py
4. Buka LM Studio dan muat model Gemma-3-4B-IT.
5. Aktifkan Local Server pada port 1234.

# Clone repository
[git clone https://github.com/username/rag-akademik-pdf.git](https://github.com/ajipangestu14072001/uas-kecerdasan-buatan)

# Install dependencies
pip install langchain langchain-community langchain-huggingface chromadb pypdf requests

## Penjelasan Fungsi Teknis
1. GemmaLLM(LLM) - Custom Wrapper
   Kelas ini menghubungkan LangChain dengan API lokal LM Studio.

   - Repetition Penalty (1.1): Secara teknis mencegah model mengulang frasa yang sama dalam satu respons.

   - Stop Sequences: Menggunakan token seperti Pertanyaan: dan \n\n untuk memutus generasi teks segera setelah jawaban selesai, mencegah "hallucination loop".

   - Timeout & Error Handling: Dilengkapi dengan requests timeout untuk memastikan aplikasi tidak berhenti (hang) jika inferensi lokal melambat.

2. answer_question() - Logic Core
   Fungsi ini adalah otak dari sistem RAG:

   - Deduplikasi Konteks: Menggunakan mekanisme seen_contents = set() untuk memastikan bahwa jika database vektor mengembalikan potongan teks yang identik, hanya satu yang dikirim ke LLM.

   - Reference Tracking: Mengekstrak metadata halaman dari objek Document untuk memberikan transparansi sumber jawaban kepada pengguna.

## Contoh Pertanyaan
- Apa syarat minimal SKS untuk mengambil skripsi?
- Bagaimana prosedur cuti akademik?

## Catatan
Jawaban dihasilkan berdasarkan isi dokumen PDF
dan disertai referensi halaman untuk mencegah halusinasi.
