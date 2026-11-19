RERANKER = "BAAI/bge-reranker-v2-m3"
reranker = []


def rerank_chunks(question, docs):
    if not reranker:
        from sentence_transformers import CrossEncoder
        reranker.append(CrossEncoder(RERANKER, max_length=1024))
    scores = reranker[0].predict([(question, doc.page_content) for doc in docs])
    return [doc for _, doc in sorted(zip(scores, docs), key=lambda pair: -pair[0])]
