from langchain_core.documents import Document
from steps.clean.blocks import remove_blocks
from steps.clean.inline import (
    links_to_text,
    remove_html_noise,
    remove_images,
    remove_markdown_noise,
)
from steps.clean.pages import drop_app_pages, drop_copies, drop_empty_pages, move_head
from steps.clean.repeated_lines import remove_repeated_lines
from steps.clean.tidy import tidy


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
    "markdown_noise": each_page(remove_markdown_noise),
    "html_noise": each_page(remove_html_noise),
    "blocks": remove_blocks,
    "repeated_lines": remove_repeated_lines,
    "tidy": each_page(tidy),
    "head": move_head,
    "app_pages": drop_app_pages,
    "empty_pages": drop_empty_pages,
    "copies": drop_copies,
}


def clean(docs, names):
    for name in names:
        docs = CLEANERS[name](docs)
    return docs
