import re


EFFECTIVE = re.compile(r"(?:effective|as of)\D{0,25}?(\d{1,2})/(\d{1,2})/(\d{2,4})", re.I)


def effective_date(doc):
    found = EFFECTIVE.search(doc.page_content)
    if not found:
        return None
    month, day, year = found.groups()
    year = int(year) + 2000 if len(year) == 2 else int(year)
    return f"{year:04d}-{int(month):02d}-{int(day):02d}"


def reorder_by_date(docs):
    dated = [n for n, doc in enumerate(docs) if effective_date(doc)]
    ordered = sorted((docs[n] for n in dated), key=effective_date, reverse=True)
    docs = list(docs)
    for n, doc in zip(dated, ordered):
        docs[n] = doc
    return docs
