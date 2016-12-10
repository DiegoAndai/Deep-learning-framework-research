"""This file generates pairs of words, groups them
by the relationship of their words and saves that
information in a JSON file"""

import json
import requests
import pickle


def generate_pairs(words, relations):

    """:parameter words: list of words to form pairs
    :parameter relations: iterable containing the relations
    by which the pairs will be constructed"""

    concept_net = "http://api.conceptnet.io/"
    all_rel_dict = {}
    for rel in relations:

        rel_dict = {}
        for word in words:

            obj = requests.get('{}query?start=/c/en/{}&rel=/r/{}'.format(concept_net, word, rel)).json()
            accounted_for = []
            for edge in obj["edges"]:
                end = edge["end"]
                end_word = end["label"]
                if end["language"] == "en" and '_' not in end["term"] and end_word != word and \
                        end_word not in accounted_for and end_word in words:
                    # This condition checks that the end or target concept
                    # of the relation is a word in English and that it is
                    # contained in the embedding.
                    accounted_for.append(end_word)
                    try:
                        rel_dict[word] += [end_word]
                    except KeyError:
                        rel_dict[word] = [end_word]
        if rel_dict:
            all_rel_dict[rel] = rel_dict

    return all_rel_dict

if __name__ == '__main__':

    path_to_embedding = "../SkipGram/"

    with open(path_to_embedding + "reverse_dictionary", "rb") as reverse_dictionary_file:
        reverse_dictionary = pickle.load(reverse_dictionary_file)

    words = list(reverse_dictionary.values())
    relations = ("Synonym", "Antonym", "Causes", "RelatedTo",
                 "FormOf", "IsA", "PartOf", "HasA")  # may add more

    with open("relation_pairs", 'w') as pf:
        json.dump(generate_pairs(words, relations), pf)


