import re

from rank_bm25 import BM25Okapi

WORD = re.compile(r"[a-z0-9]+(?:\.[0-9]+)?")
RRF_K = 60


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


RERANKER = "BAAI/bge-reranker-v2-m3"
reranker = []


def rerank(question, docs):
    if not reranker:
        from sentence_transformers import CrossEncoder
        reranker.append(CrossEncoder(RERANKER, max_length=1024))
    scores = reranker[0].predict([(question, doc.page_content) for doc in docs])
    return [doc for _, doc in sorted(zip(scores, docs), key=lambda pair: -pair[0])]


def fuse(rankings):
    scores, docs = {}, {}
    for ranking in rankings:
        for rank, doc in enumerate(ranking):
            scores[doc.id] = scores.get(doc.id, 0) + 1 / (RRF_K + rank + 1)
            docs[doc.id] = doc
    return [docs[i] for i in sorted(scores, key=scores.get, reverse=True)]


EFFECTIVE = re.compile(r"(?:effective|as of)\D{0,25}?(\d{1,2})/(\d{1,2})/(\d{2,4})", re.I)


def effective_date(doc):
    found = EFFECTIVE.search(doc.page_content)
    if not found:
        return None
    month, day, year = found.groups()
    year = int(year) + 2000 if len(year) == 2 else int(year)
    return f"{year:04d}-{int(month):02d}-{int(day):02d}"


def newest_effective_first(docs):
    dated = [n for n, doc in enumerate(docs) if effective_date(doc)]
    ordered = sorted((docs[n] for n in dated), key=effective_date, reverse=True)
    docs = list(docs)
    for n, doc in zip(dated, ordered):
        docs[n] = doc
    return docs
