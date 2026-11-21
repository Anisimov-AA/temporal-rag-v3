import hashlib
import json
import re

from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_ollama import OllamaEmbeddings
from rank_bm25 import BM25Okapi

BATCH = 64
EMBEDDINGS = {
    "bge-m3": lambda: OllamaEmbeddings(model="bge-m3", num_ctx=8192),
}
QUERY_EMBEDDINGS = {
    "bge-m3": lambda: OllamaEmbeddings(model="bge-m3", num_ctx=8192, num_gpu=0),
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


def fingerprint(docs):
    joined = "".join(sorted(doc.id for doc in docs))
    return hashlib.sha1(joined.encode("utf-8")).hexdigest()[:8]


def index_docs(chunks):
    docs = [Document(page_content=search_text(c), metadata=plain(c.metadata)) for c in chunks]
    for doc, doc_id in zip(docs, doc_ids(docs)):
        doc.id = doc_id
    return docs


def build_index(docs, name, embeddings, folder):
    ids = [doc.id for doc in docs]
    store = Chroma(collection_name=name, embedding_function=embeddings, persist_directory=str(folder))
    if set(store.get(include=[])["ids"]) == set(ids):
        return store, "reused"
    store.reset_collection()
    for start in range(0, len(docs), BATCH):
        store.add_documents(docs[start : start + BATCH], ids=ids[start : start + BATCH])
    return store, "built"


def open_index(name, embeddings, folder):
    return Chroma(collection_name=name, embedding_function=embeddings, persist_directory=str(folder))


WORD = re.compile(r"[a-z0-9]+(?:\.[0-9]+)?")


def words(text):
    return WORD.findall(text.lower())


class Keywords:
    def __init__(self, docs):
        self.docs = docs
        self.bm25 = BM25Okapi([words(doc.page_content) for doc in docs])

    def search(self, question, k):
        scores = self.bm25.get_scores(words(question))
        best = sorted(range(len(self.docs)), key=lambda i: scores[i], reverse=True)[:k]
        return [self.docs[i] for i in best]
