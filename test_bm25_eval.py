from rank_bm25 import BM25Okapi

from agent_tool_demo import (
    load_knowledge_chunks,
    tokenize_for_bm25,
)
from test_knowledge_search import TEST_CASES


TOP_K = 3


def main():
    # 与 Embedding 实验使用相同的文本块
    chunks = load_knowledge_chunks(max_chars=250)

    if not chunks:
        print("知识库为空。")
        return

    # 对文档进行分词
    valid_chunks = []
    tokenized_corpus = []

    for chunk in chunks:
        tokens = tokenize_for_bm25(chunk["text"])

        if tokens:
            valid_chunks.append(chunk)
            tokenized_corpus.append(tokens)

    bm25 = BM25Okapi(tokenized_corpus)

    total_passed = 0

    for case in TEST_CASES:
        query = case["query"]
        query_tokens = tokenize_for_bm25(query)

        scores = bm25.get_scores(query_tokens)
        query_token_set = set(query_tokens)

        # 与现有 search_knowledge 保持一致：
        # 无共同词项的文本块不进入候选结果
        results = []

        for chunk, doc_tokens, score in zip(
            valid_chunks,
            tokenized_corpus,
            scores,
        ):
            if query_token_set.isdisjoint(doc_tokens):
                continue

            item = chunk.copy()
            item["score"] = float(score)
            results.append(item)

        results.sort(
            key=lambda item: item["score"],
            reverse=True,
        )

        results = results[:TOP_K]

        # 动态识别相关文本块，与 Embedding 测试使用相同规则
        relevant_ids = {
            chunk["chunk_id"]
            for chunk in chunks
            if any(
                phrase.lower() in chunk["text"].lower()
                for phrase in case["relevant_phrases"]
            )
        }

        retrieved_ids = {
            item["chunk_id"] for item in results
        }

        hit_ids = retrieved_ids & relevant_ids

        precision = len(hit_ids) / TOP_K
        recall = (
            len(hit_ids) / len(relevant_ids)
            if relevant_ids else 0.0
        )

        retrieved_text = "\n".join(
            item["text"] for item in results
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

        for rank, item in enumerate(results, start=1):
            print(
                f"  排名={rank}, "
                f"块编号={item['chunk_id']}, "
                f"BM25分数={item['score']:.4f}"
            )

    print(
        f"\nBM25 测试汇总："
        f"{total_passed}/{len(TEST_CASES)} 通过"
    )


if __name__ == "__main__":
    main()