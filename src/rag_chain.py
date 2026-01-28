from langchain_core.documents import Document
from langchain_core.language_models import LLM
from langchain_core.prompts import PromptTemplate
from langchain_community.vectorstores import Chroma
from typing import List, Tuple, Optional
import requests

class GemmaLLM(LLM):
    model_name: str = "google/gemma-3-4b"
    base_url: str = "http://localhost:1234/v1"
    temperature: float = 0.3

    @property
    def _llm_type(self) -> str:
        return "gemma"

    def _call(self, prompt: str, stop: Optional[List[str]] = None) -> str:
        payload = {
            "model": self.model_name,
            "prompt": prompt,
            "temperature": self.temperature,
            "max_new_tokens": 150,
            "repetition_penalty": 1.1,
            "stop": ["Pertanyaan:", "User:", "Sistem:", "\n\n"]
        }

        try:
            response = requests.post(f"{self.base_url}/completions", json=payload, timeout=60)
            response.raise_for_status()
            res_json = response.json()

            if "choices" in res_json and len(res_json["choices"]) > 0:
                text = res_json["choices"][0].get("text", "").strip()
                return text
            return "Maaf, informasi tidak ditemukan."
        except Exception as e:
            return f"Error: {str(e)}"

def answer_question(vector_store: Chroma, question: str) -> Tuple[str, List[Document]]:
    raw_docs = vector_store.similarity_search(query=question, k=3)
    seen_contents = set()
    unique_docs = []
    for doc in raw_docs:
        if doc.page_content not in seen_contents:
            unique_docs.append(doc)
            seen_contents.add(doc.page_content)

    context = "\n\n".join([d.page_content for d in unique_docs])

    prompt_template = """Sistem: Kamu adalah asisten akademik kampus.

Konteks:
{context}

Pertanyaan: {question}

Jawaban Lengkap:"""

    prompt = PromptTemplate(
        input_variables=["context", "question"],
        template=prompt_template
    ).format(context=context, question=question)

    llm = GemmaLLM()
    answer = llm.invoke(prompt)

    return answer, unique_docs