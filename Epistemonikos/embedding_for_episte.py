import tensorflow as tf
import json

from open_documents import SentenceReader

with open("documents_array.json", "r") as json_file:
    loaded = json.load(json_file)

reader = SentenceReader(loaded)

words = []
i = 0
for abstract in reader:
    if abstract:
        words += abstract
    if i % 50000 == 0:
        print(i)
    i += 1

print("Data size", len(words))
