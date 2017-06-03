import json
from collections import namedtuple

WordTuple = namedtuple("WordTuple", ["ps_proba", "sr_proba"])

def get_word_proba(word, string):
    if word in string:
        return 1
    return 0

def get_class_word_proba(word, set_of_strings):
    total_documents = len(set_of_strings)
    sum_of_appareances = sum(map(lambda d: get_word_proba(word, d), set_of_strings))
    return sum_of_appareances/total_documents


with open("../Set4/train_papers", "r") as paper_train_file, \
     open("../Set4/test_papers", "r") as paper_test_file:
    train = json.load(paper_train_file)
    test = json.load(paper_test_file)

    total = train + test

#####CREATE DICTIONARY
print("creating dictionary")

words = {}

all_abstracts = (paper["abstract"] for paper in total)

for paper in total:
    listed_words = paper["abstract"].split()
    check_occurence_sr = False
    check_occurence_ps = False
    for word in listed_words:
        if word not in words:
            words[word] = {"systematic-review": 0, "primary-study": 0}

        if paper["classification"] == "systematic-review" and not check_occurence_sr:
            words[word]["systematic-review"] += 1
            check_occurence_sr = True

        if paper["classification"] == "primary-study" and not check_occurence_ps:
            words[word]["primary-study"] += 1
            check_occurence_ps = True


print("dictionary done, len: ", len(words))
######CALCULATE PROBABILITIES PER CLASS
print("calculating probabilities")

results = {}

ps_total = sum((1 if paper["classification"] == "primary-study" else 0 for paper in total))
sr_total = sum((1 if paper["classification"] == "systematic-review" else 0 for paper in total))

i = 0
for word, ocurrence_dict in words.items():
    ps_proba = words[word]["primary-study"] / ps_total
    sr_proba = words[word]["systematic-review"] / sr_total
    results[word] = {"ps_proba": ps_proba, "sr_proba": sr_proba}

    if i % 10000 == 0:
        print("{}/{}".format(i, len(words)))

    i+= 1

######SAVE RESULTS


with open("proba_results.json", "w") as json_out:
    json.dump(results, json_out)
