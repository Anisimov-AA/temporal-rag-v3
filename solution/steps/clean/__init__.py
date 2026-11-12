from langchain_core.documents import Document
from steps.clean.inline import links_to_text, remove_images


def each_page(fix):
    def step(docs):
        return [
            Document(page_content=fix(doc.page_content), metadata=doc.metadata)
            for doc in docs
        ]

    return step


CLEANERS = {
    "images": each_page(remove_images),
    "links": each_page(links_to_text),
}


def clean(docs, names):
    for name in names:
        docs = CLEANERS[name](docs)
    return docs
