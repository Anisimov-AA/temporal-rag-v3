import hashlib
import json

from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_ollama import OllamaEmbeddings

BATCH = 64
EMBEDDINGS = {
    "bge-m3": lambda: OllamaEmbeddings(model="bge-m3", num_ctx=8192),
}


def search_text(chunk):
    m = chunk.metadata
    path = " › ".join(x for x in (m.get("h1"), m.get("h2")) if x)
    return "\n".join(x for x in (m["title"], path, chunk.page_content) if x)


def plain(metadata):
    return {
        key: ", ".join(value) if isinstance(value, list) else value
        for key, value in metadata.items()
        if value is not None
    }


def doc_ids(docs):
    ids, seen = [], {}
    for doc in docs:
        key = json.dumps([doc.page_content, doc.metadata], sort_keys=True)
        digest = hashlib.sha1(key.encode("utf-8")).hexdigest()
        seen[digest] = seen.get(digest, 0) + 1
        ids.append(f"{digest}-{seen[digest]}")
    return ids


def build_index(chunks, name, embeddings, folder):
    docs = [Document(page_content=search_text(c), metadata=plain(c.metadata)) for c in chunks]
    ids = doc_ids(docs)
    store = Chroma(collection_name=name, embedding_function=embeddings, persist_directory=str(folder))
    if set(store.get(include=[])["ids"]) == set(ids):
        return store, "reused"
    store.reset_collection()
    for start in range(0, len(docs), BATCH):
        store.add_documents(docs[start : start + BATCH], ids=ids[start : start + BATCH])
    return store, "built"
