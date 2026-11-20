import re

COPY = re.compile(r"-(\d+)$")


def page_series(url):
    return COPY.sub("", url.rstrip("/"))


def page_version(doc):
    copy = COPY.search(doc.metadata["url"].rstrip("/"))
    return doc.metadata.get("published") or "", int(copy.group(1)) if copy else 1


def drop_old_versions(docs):
    newest = {}
    for doc in docs:
        series = page_series(doc.metadata["url"])
        if series not in newest or page_version(doc) > page_version(newest[series]):
            newest[series] = doc
    keep = {newest[s].metadata["url"] for s in newest}
    return [doc for doc in docs if doc.metadata["url"] in keep]
