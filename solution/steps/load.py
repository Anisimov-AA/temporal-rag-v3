from langchain_core.documents import Document


def to_documents(snapshot):
    docs = []
    for page in snapshot.pages:
        docs.append(
            Document(
                page_content=page.markdown,
                metadata={
                    "url": page.url,
                    "title": page.title,
                    "fetched_at": page.fetched_at,
                    "snapshot_id": snapshot.snapshot_id,
                    "taken_on": snapshot.taken_on,
                },
            )
        )
    return docs
