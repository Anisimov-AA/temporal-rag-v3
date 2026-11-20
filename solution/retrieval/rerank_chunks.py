RERANKER = "BAAI/bge-reranker-v2-m3"
reranker = []


def rerank_chunks(question, docs):
    if not reranker:
        import torch
        from sentence_transformers import CrossEncoder
        model = CrossEncoder(RERANKER, max_length=1024)
        if torch.cuda.is_available():
            model.model.half()
        reranker.append(model)
    scores = reranker[0].predict([(question, doc.page_content) for doc in docs])
    return [doc for _, doc in sorted(zip(scores, docs), key=lambda pair: -pair[0])]
