from agent_tool_demo import load_knowledge_chunks


def main():
    chunks = load_knowledge_chunks(max_chars=250)

    print(f"共切分出 {len(chunks)} 个文本块。")

    for chunk in chunks:
        print(
            f"\n--- {chunk['source']} "
            f"/ chunk {chunk['chunk_id']} ---"
        )
        print(chunk["text"])


if __name__ == "__main__":
    main()