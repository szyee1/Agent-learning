from agent_tool_demo import (
    search_knowledge,
    load_knowledge_chunks,
)


TEST_CASES = [
    {
        "name": "知识库工具能力",
        "query": "目前实现了哪些知识库读取和检索工具",
        "expected_terms": [
            "list_knowledge_files",
            "read_knowledge_file",
            "search_knowledge",
        ],
        "relevant_phrases": [
            "list_knowledge_files",
            "read_knowledge_file",
            "search_knowledge",
        ]
    },
    {
        "name": "关键词检索的局限",
        "query": "当前关键词检索有哪些局限",
        "expected_terms": [
            "同义表达",
            "语义相近",
            "Embedding",
        ],
        "relevant_phrases": [
            "同义表达",
            "语义相近",
            "Embedding",
        ]
    },
    {
        "name": "后续开发计划",
        "query": "后续准备实现哪些检索能力",
        "expected_terms": [
            "Embedding",
            "检索排序",
            "召回率",
        ],
        "relevant_phrases": [
            "实现基于 Embedding 的语义检索",
            "检索排序",
            "召回率",
        ]
    },
    {
    "name": "语义改写检索",
    "query": "我想知道这个系统能够对资料库执行哪些操作",
    "expected_terms": [
        "list_knowledge_files",
        "read_knowledge_file",
        "search_knowledge",
    ],
    "relevant_phrases": [
        "list_knowledge_files",
        "read_knowledge_file",
        "search_knowledge",
    ],
},
]


def main():
    passed = 0

    for case in TEST_CASES:
        results = search_knowledge(
            case["query"],
            top_k=5,
        )

        retrieved_text = "\n".join(
            item["text"] for item in results
        )

        matched_terms = [
            term
            for term in case["expected_terms"]
            if term.lower() in retrieved_text.lower()
        ]

        success = len(matched_terms) == len(case["expected_terms"])

        # 当前检索结果中的文本块编号
        retrieved_ids = {
            item["chunk_id"] for item in results
        }

        # 根据当前文档和当前切分方式，动态识别相关文本块
        all_chunks = load_knowledge_chunks(max_chars=250)

        relevant_phrases = case["relevant_phrases"]

        relevant_ids = {
            chunk["chunk_id"]
            for chunk in all_chunks
            if any(
                phrase.lower() in chunk["text"].lower()
                for phrase in relevant_phrases
            )
        }

        # 本次实际检索命中的相关文本块
        hit_ids = retrieved_ids & relevant_ids

        # Precision@K：检索结果中相关文本块的比例
        k = 5

        # 固定 K 的 Precision@5
        precision_at_k = len(hit_ids) / k

        # 实际返回结果中的相关比例
        precision_returned = (
            len(hit_ids) / len(retrieved_ids)
            if retrieved_ids else 0.0
        )

        # Recall@K：找回了多少已标注的相关文本块
        recall = (
            len(hit_ids) / len(relevant_ids)
            if relevant_ids else 0.0
        )

        if success:
            passed += 1

        print(f"\n测试：{case['name']}")
        print(f"问题：{case['query']}")
        print(f"检索到的文本块：{len(results)}")
        print(f"命中的预期关键词：{matched_terms}")
        print(f"相关文本块：{sorted(relevant_ids)}")
        print(f"命中的相关文本块：{sorted(hit_ids)}")
        print(f"Precision@5：{precision_at_k:.2%}")
        print(f"实际返回结果的相关比例：{precision_returned:.2%}")
        print(f"Recall@5：{recall:.2%}")
        print(f"结果：{'通过' if success else '失败'}")

        for item in results:
            print(
                f"  来源={item['source']}, "
                f"块编号={item['chunk_id']}, "
                f"分数={item['score']}"
            )

    print(f"\n测试汇总：{passed}/{len(TEST_CASES)} 通过")


if __name__ == "__main__":
    main()