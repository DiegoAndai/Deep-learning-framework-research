import string
import json
import pickle
import os
import io
from classify import DocumentSpace

log = open("full_pipeline_log.txt", 'a')  # save runtime information here

path_to_docs = ""
path_to_embeddings_folder = ""


# open original documents' file:
with open(path_to_docs, encoding="utf-8") as docs_json:
    documents = json.load(docs_json)


# clean abstracts:
def parse_abstract(abstract):
    abstract = abstract.lower()
    words = abstract.split()
    clean = [parse_word(word) for word in words]
    return list(filter(lambda l: l, clean))

keep = string.ascii_lowercase + '-'
dismiss = "123456789"


def parse_word(word):
    for ch in dismiss:
        if ch in word:
            return None
    return "".join(ch for ch in word if ch in keep)


for doc in documents:
    abstract = doc["abstract"]
    if abstract:
        doc["abstract"] = parse_abstract(abstract)

# The docs now have their abstracts parsed


# work only with SRs and PSs:
docs_sr_ps = [doc for doc in filter(lambda d: d["classification"] in ("primary-study", "systematic-review"), documents)]

# 10-fold cross validation's folds by years (2002 to 2011):
for year in range(2002, 2012):

    if not os.path.isfile("fold_{}.json".format(year)):  # if the fold doesn't exist as a file
        year_docs = [d for d in filter(lambda doc: str(year) in doc["year"], docs_sr_ps)]
        other_docs = [d for d in filter(lambda doc: str(year) not in doc["year"], docs_sr_ps)]
        with io.open("fold_{}.json".format(year), "w", encoding="utf-8") as fold_file:
            json.dump({"{} docs".format(year): year_docs, "other docs": other_docs}, fold_file, ensure_ascii=False)
    else:  # if it exists as a file
        with open("fold_{}.json".format(year), encoding="utf-8") as fold_file:
            fold_docs = json.load(fold_file)
            year_docs = fold_docs["{} docs".format(year)]
            other_docs = fold_docs["other docs"]

    # classify with each fold


# work with every category (except uncategorised)

log.close()
