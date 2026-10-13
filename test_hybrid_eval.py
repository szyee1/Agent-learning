from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer

from agent_tool_demo import (
    load_knowledge_chunks,
    tokenize_for_bm25,
)
from test_knowledge_search import TEST_CASES


MODEL_NAME = "BAAI/bge-small-zh-v1.5"
QUERY_INSTRUCTION = "为这个句子生成表示以用于检索相关文章："

TOP_K = 3
RRF_K = 60


def main():
    # 1. 加载统一的知识库和文本块
    chunks = load_knowledge_chunks(max_chars=250)

    if not chunks:
        print("知识库为空。")
        return

    documents = [item["text"] for item in chunks]

    # 2. 建立 BM25 检索模型
    valid_chunks = []
    tokenized_corpus = []

    for chunk in chunks:
        tokens = tokenize_for_bm25(chunk["text"])

        if tokens:
            valid_chunks.append(chunk)
            tokenized_corpus.append(tokens)

    bm25 = BM25Okapi(tokenized_corpus)

    # 3. 加载 Embedding 模型并计算文档向量
    print("正在加载 Embedding 模型...")
    model = SentenceTransformer(MODEL_NAME)

    document_vectors = model.encode(
        documents,
        normalize_embeddings=True,
        convert_to_numpy=True,
        show_progress_bar=True,
    )

    total_passed = 0

    for case in TEST_CASES:
        query = case["query"]

        # ---------- BM25 排名 ----------
        query_tokens = tokenize_for_bm25(query)
        query_token_set = set(query_tokens)

        bm25_scores = bm25.get_scores(query_tokens)
        bm25_results = []

        for chunk, doc_tokens, score in zip(
            valid_chunks,
            tokenized_corpus,
            bm25_scores,
        ):
            if query_token_set.isdisjoint(doc_tokens):
                continue

            bm25_results.append(
                (chunk, float(score))
            )

        bm25_results.sort(
            key=lambda item: item[1],
            reverse=True,
        )

        # ---------- Embedding 排名 ----------
        query_vector = model.encode(
            QUERY_INSTRUCTION + query,
            normalize_embeddings=True,
            convert_to_numpy=True,
        )

        embedding_scores = document_vectors @ query_vector

        embedding_results = sorted(
            zip(chunks, embedding_scores),
            key=lambda item: float(item[1]),
            reverse=True,
        )

        # ---------- RRF 排名融合 ----------
        rrf_scores = {}
        chunk_by_id = {
            chunk["chunk_id"]: chunk
            for chunk in chunks
        }

        for ranked_results in (
            bm25_results,
            embedding_results,
        ):
            for rank, (chunk, _) in enumerate(
                ranked_results,
                start=1,
            ):
                chunk_id = chunk["chunk_id"]

                rrf_scores[chunk_id] = (
                    rrf_scores.get(chunk_id, 0.0)
                    + 1.0 / (RRF_K + rank)
                )

        hybrid_results = sorted(
            rrf_scores.items(),
            key=lambda item: item[1],
            reverse=True,
        )[:TOP_K]

        retrieved_chunks = [
            chunk_by_id[chunk_id]
            for chunk_id, _ in hybrid_results
        ]

        retrieved_ids = {
            chunk["chunk_id"]
            for chunk in retrieved_chunks
        }

        # 使用已有的预期短语动态标记相关文本块
        relevant_ids = {
            chunk["chunk_id"]
            for chunk in chunks
            if any(
                phrase.lower() in chunk["text"].lower()
                for phrase in case["relevant_phrases"]
            )
        }

        hit_ids = retrieved_ids & relevant_ids

        precision = len(hit_ids) / TOP_K
        recall = (
            len(hit_ids) / len(relevant_ids)
            if relevant_ids else 0.0
        )

        retrieved_text = "\n".join(
            chunk["text"]
            for chunk in retrieved_chunks
        )

        matched_terms = [
            term
            for term in case["expected_terms"]
            if term.lower() in retrieved_text.lower()
        ]

        passed = (
            len(matched_terms) == len(case["expected_terms"])
        )

        if passed:
            total_passed += 1

        print(f"\n测试：{case['name']}")
        print(f"问题：{query}")
        print(f"命中的预期关键词：{matched_terms}")
        print(f"相关文本块：{sorted(relevant_ids)}")
        print(f"命中的相关文本块：{sorted(hit_ids)}")
        print(f"Precision@{TOP_K}：{precision:.2%}")
        print(f"Recall@{TOP_K}：{recall:.2%}")
        print(f"结果：{'通过' if passed else '失败'}")

        for rank, (chunk_id, score) in enumerate(
            hybrid_results,
            start=1,
        ):
            print(
                f"  排名={rank}, "
                f"块编号={chunk_id}, "
                f"RRF分数={score:.6f}"
            )

    print(
        f"\n混合检索测试汇总："
        f"{total_passed}/{len(TEST_CASES)} 通过"
    )


if __name__ == "__main__":
    main()