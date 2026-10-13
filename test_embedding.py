import numpy as np
from sentence_transformers import SentenceTransformer

from agent_tool_demo import load_knowledge_chunks


# 使用中文 Embedding 模型
MODEL_NAME = "BAAI/bge-small-zh-v1.5"

# BGE 模型针对短查询检索文档的推荐指令
QUERY_INSTRUCTION = "为这个句子生成表示以用于检索相关文章："


def main():
    # 1. 加载现有知识库，沿用当前 250 字符切分设置
    chunks = load_knowledge_chunks(max_chars=250)

    if not chunks:
        print("知识库为空，没有可检索的文本块。")
        return

    documents = [chunk["text"] for chunk in chunks]

    # 2. 加载模型
    print(f"正在加载 Embedding 模型：{MODEL_NAME}")
    model = SentenceTransformer(MODEL_NAME)
    print("模型加载成功。")

    # 3. 将文档转换为向量
    document_embeddings = model.encode(
        documents,
        normalize_embeddings=True,
        convert_to_numpy=True,
        show_progress_bar=True,
    )

    # 4. 使用一个不同表达方式的问题进行测试
    query = "我想知道这个系统能够对资料库执行哪些操作。"

    query_embedding = model.encode(
        QUERY_INSTRUCTION + query,
        normalize_embeddings=True,
        convert_to_numpy=True,
    )

    print(f"\n查询问题：{query}")
    print(f"文本块数量：{len(chunks)}")
    print(f"每个向量的维度：{query_embedding.shape[0]}")

    # 5. 计算查询向量与所有文档向量的相似度
    # 向量已归一化，因此点积等于余弦相似度
    scores = document_embeddings @ query_embedding

    # 6. 按相似度从高到低排序
    ranked_results = sorted(
        zip(chunks, scores),
        key=lambda item: float(item[1]),
        reverse=True,
    )

    print("\n===== Embedding 检索结果 =====")

    for rank, (chunk, score) in enumerate(
        ranked_results[:5],
        start=1,
    ):
        print(f"\n排名：{rank}")
        print(f"文件：{chunk['source']}")
        print(f"文本块编号：{chunk['chunk_id']}")
        print(f"相似度：{float(score):.4f}")
        print(f"内容：{chunk['text'][:200]}")


if __name__ == "__main__":
    main()