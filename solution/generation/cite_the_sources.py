from contract import Source

MIN_SOURCES = 2
MAX_SOURCES = 8


def cite_the_sources(chunks, pages):
    if not pages:
        pages = range(1, len(chunks) + 1)
    cited = [chunks[p - 1] for p in pages if 1 <= p <= len(chunks)]
    sources, urls = [], []
    for chunk in cited + list(chunks):
        url = chunk.metadata["url"]
        if url in urls or (chunk not in cited and len(urls) >= MIN_SOURCES):
            continue
        urls.append(url)
        sources.append(Source(url=url, snapshot_id=chunk.metadata.get("snapshot_id")))
    return sources[:MAX_SOURCES]
