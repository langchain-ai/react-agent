import json

import bm25s
import chromadb
from langchain_huggingface import HuggingFaceEmbeddings
from Stemmer import Stemmer


def prepare_hybrid_indices(file_path="data/error_definitions.json"):
    # 1. 데이터 로드
    with open(file_path, "r", encoding="utf-8") as f:
        error_definitions = json.load(f)

    # --- [Part A: ChromaDB - Vector Index] ---
    client = chromadb.PersistentClient(path="./chroma_db")
    embed_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    collection = client.get_or_create_collection(name="airflow_errors")

    # --- [Part B: BM25S - Lexical Index] ---
    # BM25S 검색 대상이 될 코퍼스(Corpus) 생성
    corpus = []
    metadata_list = []

    for item in error_definitions:
        # 검색 품질을 높이기 위해 ID, 설명, 예시를 합친 텍스트 생성
        combined_text = f"{item['error_id']} {item['description']} " + " ".join(
            item.get("representative_examples", [])
        )
        corpus.append(combined_text)

        # 메타데이터 저장 (나중에 결과 매칭용)
        metadata_list.append(
            {
                "error_id": item["error_id"],
                "category": item["category"],
                "resolution": item["resolution_step"],
            }
        )

        # ChromaDB에도 동일하게 추가
        collection.upsert(
            documents=[combined_text],
            metadatas=[metadata_list[-1]],
            ids=[item["error_id"]],
        )

    # BM25S 토큰화 및 인덱싱
    stemmer = Stemmer("english")
    corpus_tokens = bm25s.tokenize(corpus, stemmer=stemmer)

    bm25_model = bm25s.BM25()
    bm25_model.index(corpus_tokens)

    # BM25S 인덱스 로컬 저장 (중요: metadata와 함께 저장)
    bm25_model.save("./bm25s_index", corpus=corpus)

    # 매칭을 위한 메타데이터 별도 저장
    with open("./bm25s_index/metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata_list, f, indent=2)

    print("✅ Hybrid Indices (ChromaDB + BM25S) Prepared!")


if __name__ == "__main__":
    prepare_hybrid_indices()
