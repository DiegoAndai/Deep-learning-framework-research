from open_documents import PaperReader
import collections
import os
import string
import pickle


n = input("enter 'w' to load Epistemonikos data from words "
          "file or just enter to parse Epistemonikos data from json file ").lower()

if os.path.exists('words.txt') and n == "w":
    with open('words.txt') as w_file:
        episte_words = [word.rstrip() for word in w_file]

elif not n:
    with open("documents_array.json", "r") as json_file:
        loaded = json.load(json_file)

    reader = PaperReader(loaded)
    print("generating words")
    reader.generate_words_list()
    reader.save_words()
    episte_words = reader.words

#open_wiki
keep = string.ascii_lowercase + '-'
with open("allwiki", "r", encoding="utf-8") as allwiki:
    wiki_words = []
    for line in allwiki:
        line = line.lower()
        line_words = line.split()
        for word in line_words:
            valid = True
            for ch in "123456789</>":
                if ch in word:
                    valid = False
            if valid:
                wiki_words.append("".join(ch for ch in word if ch in keep))

print('Wiki size', len(wiki_words), 'Episte size', len(episte_words))

# Step 2: Build the dictionary and replace rare words with UNK token.
print("step 2")
vocabulary_size = 50000

count = [['UNK', -1]]  # sub_list ['UNK', -1] is used to count uncommon words.
count.extend(collections.Counter(wiki_words + episte_words).most_common(vocabulary_size - 1))  # List of tuples of words with
print("count ready")
# their appearances. The parameter of most_common determines how many most common words it will return.
dictionary = {t[1] : t[0] for t in enumerate((t[0] for t in count))}
reverse_dictionary = dict(zip(dictionary.values(), dictionary.keys()))  # to get the word by its id.
print("dictionaries ready")

with open("reverse_dictionary", "wb") as _file:
    pickle.dump(reverse_dictionary, _file)

with open("dictionary", "wb") as _file:
    pickle.dump(dictionary, _file)

print("dictionaries saved")

def build_dataset(words, dictionary):
    count = [['UNK', -1]]
    counter = collections.Counter(words)
    i = 0
    print("building count")
    for word in dictionary:
        if word != "UNK":
            count.append([word, counter[word]])
        i += 1
        if i % 10000 == 0:
            print(i)
    print("build count ready")
    data = list()  # original list of words, but with id's instead of words.
    unk_count = 0  # appearances of uncommon words
    i = 0
    print("building data")
    for word in words:
        if word in dictionary:  # if it is common enough
            index = dictionary[word]  # get it's id
        else:
            index = 0  # dictionary['UNK']
            unk_count += 1
        data.append(index)
        i += 1
        if i % 10000000 == 0:
            print(i)
    count[0][1] = unk_count  # update uncommon appearances.
    count.sort(key = lambda t: t[1])
    print("build ready")
    return data, count


print("building wiki")
wiki_data, wiki_count = build_dataset(wiki_words, dictionary)
print("wiki ready")
print("building episte")
episte_data, episte_count = build_dataset(episte_words, dictionary)
print("episte ready")

print("saving files")

with open("wiki_data", "wb") as _file:
    pickle.dump(wiki_data, _file)

with open("wiki_count", "wb") as _file:
    pickle.dump(wiki_count, _file)


with open("episte_data", "wb") as _file:
    pickle.dump(episte_data, _file)

with open("episte_count", "wb") as _file:
    pickle.dump(episte_count, _file)

print("files saved")
