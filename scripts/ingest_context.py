from my_llm.context_store import ContextStore


if __name__ == "__main__":
    store = ContextStore()
    total = store.ingest_directory()
    print(f"Indexed chunks: {total}")
