RRF_K = 60


def fuse(rankings):
    scores, docs = {}, {}
    for ranking in rankings:
        for rank, doc in enumerate(ranking):
            scores[doc.id] = scores.get(doc.id, 0) + 1 / (RRF_K + rank + 1)
            docs[doc.id] = doc
    return [docs[i] for i in sorted(scores, key=scores.get, reverse=True)]


def retrieve_chunks(store, keywords, question, k, search):
    lists = []
    if "vectors" in search:
        lists.append(store.similarity_search(question, k=k))
    if "bm25" in search:
        lists.append(keywords.search(question, k))
    return fuse(lists)
