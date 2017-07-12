import json
import string
import io
from collections import OrderedDict


def clean_word(w):
    return w.strip(string.punctuation).lower()


def has_numbers(w):
    for ch in w:
        if ch in string.digits:
            return True
    return False


with open("documents_array.json", encoding="utf-8") as jf:
    papers = json.load(jf)

freq_dict = {"primary-study": {}, "systematic-review": {}, "both": {}}
count = 0
for paper in papers:
    if paper["classification"] == "primary-study" or paper["classification"] == "systematic-review":
        if paper["abstract"]:
            for word in paper["abstract"].split():
                word = clean_word(word)
                if word and not has_numbers(word):
                    if word in freq_dict[paper["classification"]]:
                        freq_dict[paper["classification"]][word] += 1
                        freq_dict["both"][word] += 1
                    else:
                        freq_dict[paper["classification"]].update({word: 1})
                        if word in freq_dict["both"]:
                            freq_dict["both"][word] += 1
                        else:
                            freq_dict["both"].update({word: 1})
    count += 1
    if not count % 25000:
        print("Processed {} papers".format(count))

with io.open("word_frequency.json", "w", encoding="utf-8") as wf:
    json.dump({"primary-study": OrderedDict(sorted(freq_dict["primary-study"].items(), key=lambda t: t[1])),
               "systematic-review": OrderedDict(sorted(freq_dict["systematic-review"].items(), key=lambda t: t[1])),
               "both": OrderedDict(sorted(freq_dict["both"].items(), key=lambda t: t[1]))}, wf, ensure_ascii=False)
