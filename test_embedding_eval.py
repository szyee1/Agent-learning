from sentence_transformers import SentenceTransformer

from agent_tool_demo import load_knowledge_chunks
from test_knowledge_search import TEST_CASES


MODEL_NAME = "BAAI/bge-small-zh-v1.5"
QUERY_INSTRUCTION = "为这个句子生成表示以用于检索相关文章："
TOP_K = 3


def main():
    # 1. 加载同一份知识库
    chunks = load_knowledge_chunks(max_chars=250)

    if not chunks:
        print("知识库为空。")
        return

    documents = [item["text"] for item in chunks]

    # 2. 加载模型，只加载一次
    print("正在加载 Embedding 模型...")
    model = SentenceTransformer(MODEL_NAME)

    # 3. 预先计算所有文档的向量
    document_vectors = model.encode(
        documents,
        normalize_embeddings=True,
        convert_to_numpy=True,
        show_progress_bar=True,
    )

    total_passed = 0

    # 4. 依次评测已有的三个测试案例
    for case in TEST_CASES:
        query = case["query"]

        query_vector = model.encode(
            QUERY_INSTRUCTION + query,
            normalize_embeddings=True,
            convert_to_numpy=True,
        )

        # 5. 计算相似度并排序
        scores = document_vectors @ query_vector

        ranked = sorted(
            zip(chunks, scores),
            key=lambda pair: float(pair[1]),
            reverse=True,
        )

        results = ranked[:TOP_K]

        # 6. 根据预期短语，动态确定相关文本块
        relevant_phrases = case["relevant_phrases"]

        relevant_ids = {
            item["chunk_id"]
            for item in chunks
            if any(
                phrase.lower() in item["text"].lower()
                for phrase in relevant_phrases
            )
        }

        retrieved_ids = {
            item["chunk_id"] for item, _ in results
        }

        hit_ids = retrieved_ids & relevant_ids

        precision = len(hit_ids) / TOP_K
        recall = (
            len(hit_ids) / len(relevant_ids)
            if relevant_ids else 0.0
        )

        retrieved_text = "\n".join(
            item["text"] for item, _ in results
        )

        matched_terms = [
            term
            for term in case["expected_terms"]
            if term.lower() in retrieved_text.lower()
        ]

        success = (
            len(matched_terms) == len(case["expected_terms"])
        )

        if success:
            total_passed += 1

        print(f"\n测试：{case['name']}")
        print(f"问题：{query}")
        print(f"命中的预期关键词：{matched_terms}")
        print(f"相关文本块：{sorted(relevant_ids)}")
        print(f"命中的相关文本块：{sorted(hit_ids)}")
        print(f"Precision@{TOP_K}：{precision:.2%}")
        print(f"Recall@{TOP_K}：{recall:.2%}")
        print(f"结果：{'通过' if success else '失败'}")

        for rank, (item, score) in enumerate(results, start=1):
            print(
                f"  排名={rank}, "
                f"块编号={item['chunk_id']}, "
                f"相似度={float(score):.4f}"
            )

    print(
        f"\nEmbedding 测试汇总："
        f"{total_passed}/{len(TEST_CASES)} 通过"
    )


if __name__ == "__main__":
    main()