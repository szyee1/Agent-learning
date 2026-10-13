from agent_tool_demo import search_knowledge_embedding


def main():
    query = "我想知道这个系统能够对资料库执行哪些操作"

    results = search_knowledge_embedding(
        query,
        top_k=3,
    )

    print(f"查询问题：{query}")
    print(f"检索结果数量：{len(results)}")

    for rank, item in enumerate(results, start=1):
        print(f"\n排名：{rank}")
        print(f"文本块编号：{item['chunk_id']}")
        print(f"相似度：{item['score']}")
        print(f"内容：{item['text'][:200]}")


if __name__ == "__main__":
    main()