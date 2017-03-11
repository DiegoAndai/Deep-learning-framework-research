class MaxPoolLab:

    def __init__(self):

        self.results = dict() #save results, keys referring to document
        self.meta = dict()

    def add_document(self, doc_identifier):

        if doc_identifier in self.results:
            return("Identifier already exists")
        else:
            self.results[doc_identifier] = list()
            return("Added succesfully")

    def add_document_info(self, doc_identifier, key, value):

        if doc_identifier not in self.meta:
            self.meta[doc_identifier] = dict()
        self.meta[doc_identifier][key] = value

    def delete_document(self, doc_identifier):

        if doc_identifier in self.results:
            del self.results[doc_identifier]
            return("Deleted succesfully")
        else:
            return("Couldn't find identifier")

    def add_word_occurrence(self, word, index, doc_identifier):

        if doc_identifier in self.results:
            self.results[doc_identifier].append((index, word))
        else:
            return("Couldn't find identifier")

    def obtain_results_tuples(self, doc_identifier):

        if doc_identifier in self.results:
            return self.results[doc_identifier]
        else:
            return("Couldn't find identifier")

    def obtain_results(self, doc_identifier = None):

        processed_results = dict()
        if doc_identifier:
            result_tuples = self.obtain_results_tuples(doc_identifier)
            indexes, words = self.process_results(result_tuples)
            processed_results[doc_identifier] = {"indexes_occurrence": indexes,
                                                 "words_occurrence": words,
                                                 "meta": self.meta[doc_identifier]}
        else:
            for doc_identifier, result_tuples in self.results.items():
                indexes, words = self.process_results(result_tuples)
                processed_results[doc_identifier] = {"indexes_occurrence": indexes,
                                                     "words_occurrence": words,
                                                     "meta": self.meta[doc_identifier]}

        return processed_results

    def infographic_from_results(self, doc_identifier = None):

        results = self.obtain_results(doc_identifier)
        infographic = ""
        for _id, result in results.items():
            infographic += "\nid: {}\n".format(_id)

            sorted_by_appareance = sorted(result["indexes_occurrence"].items(),
                                          key = lambda t: t[0])
            infographic += "\n"
            total_words = 0
            total_occurrences = 0
            for _tuple in sorted_by_appareance:
                infographic += "|{}".format(_tuple[1])
                total_occurrences += _tuple[1]
                total_words += 1
            infographic += "|"

            sorted_by_occurrence = sorted(result["words_occurrence"].items(),
                                          key = lambda t: t[1])
            infographic += "\n"
            i = 1
            for _tuple in sorted_by_occurrence[:5]:
                infographic += "{}. {}\n".format(i, _tuple[0])
                i += 1

            infographic += "Total words = {}\n".format(total_words)
            infographic += "Total occurrences = {}\n".format(total_occurrences)


        return infographic

    def process_results(self, result_tuples):

        index_occurrence_dict = dict()
        word_occurrence_dict = dict()

        for _tuple in result_tuples:

            if _tuple[0] in index_occurrence_dict:
                index_occurrence_dict[_tuple[0]] += 1
            else:
                index_occurrence_dict[_tuple[0]] = 1

            if _tuple[1] in word_occurrence_dict:
                word_occurrence_dict[_tuple[1]] += 1
            else:
                word_occurrence_dict[_tuple[1]] = 1

        return index_occurrence_dict, word_occurrence_dict
