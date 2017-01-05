from Epistemonikos.PaperClassification.PVPClassification.PVPClassifier import PVPClassifier, get_n_papers
import pickle
import json

if __name__ == '__main__':

    embeddings = []
    vocabs = []
    for embed_name in ("wikiembedding", "episteembedding"):

        with open(embed_name, "rb") as embed_serialized:
            final_embeddings = pickle.load(embed_serialized)

        embeddings.append(final_embeddings)

        with open("reverse_dictionary", "rb") as reverse_dictionary_file:
            reverse_dictionary = pickle.load(reverse_dictionary_file)

        vocabs.append([reverse_dictionary[i] for i in range(len(reverse_dictionary))])

    document_types = ["systematic-review",
                      "structured-summary-of-systematic-review",
                      "primary-study",
                      "overview",
                      "structured-summary-of-primary-study"]

    with open("../SkipGram/documents_array.json", encoding="utf-8") as da:
        ref_papers = json.load(da)

    to_classify = get_n_papers(200, "../SkipGram/documents_array.json")

    wiki_class = PVPClassifier(embeddings[0], vocabs[0], document_types, ref_papers)
    wiki_class.get_ref_vectors(new_n_save=True)
    wiki_class.get_abs_vectors(to_classify, new_n_save=True)
    wiki_class.classify()

    episte_class = PVPClassifier(embeddings[1], vocabs[1], document_types, ref_papers)
    episte_class.get_ref_vectors(new_n_save=True)
    episte_class.get_abs_vectors(to_classify, new_n_save=True)
    episte_class.classify()

    print(wiki_class.get_conf_matrix(), episte_class.get_conf_matrix())
    print(wiki_class.get_accuracy(), episte_class.get_accuracy())
    print(wiki_class.recalls(), episte_class.recalls())
    print(wiki_class.precisions(), episte_class.precisions())