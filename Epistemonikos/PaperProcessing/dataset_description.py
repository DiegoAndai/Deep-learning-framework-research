from open_documents import PaperReader
import json

with open("documents_array.json", encoding="utf-8") as docs:
    reader = PaperReader(json.load(docs), filters=["systematic-review", "primary-study"], train_percent=80,
                         abstracts_min=30, even_test=True)

    print(reader.span_stats(filename = "output"))
