import string
import pickle
import json


class PaperReader:

    """Class to generate training and testing sets for word2vec language models from Epistemonikos Papers. Most
    significant methods: __init__, save_train/test_papers and generate_train_text_file."""

    def __init__(self, papers, filters=None, train_percent=100, abstracts_min=None, dispose_no_abstract=True):

        self.abstracts_min_lmt = abstracts_min  # used to dispose short abstracts

        # Dispose of short (if abstracts_min is True) or empty abstracts otherwise
        self._papers = []
        for paper in papers:
            if abstracts_min and paper["abstract"] and len(paper["abstract"].split()) >= abstracts_min:
                self._papers.append(paper)
            elif not abstracts_min and paper["abstract"] and dispose_no_abstract:
                self._papers.append(paper)
            elif not abstracts_min and not dispose_no_abstract:
                self._papers = papers
                break

        self.keep = string.ascii_lowercase + '-'
        self.dismiss = "123456789"
        self.words = []
        self.filter_by = filters if filters else []  # filter by paper classification (systematic-review,
        # structured-summary-of-systematic-review, primary-study, overview, structured-summary-of-primary-study)

        self.filtered_papers = list(map(lambda pap: {"abstract": ' '.join(self.parse_line(pap["abstract"])),
                                                     "classification": pap["classification"], "id": pap["id"]},
                                        [p for p in filter(lambda paper: paper["classification"] in self.filter_by,
                                                           self._papers)] if self.filter_by else self._papers))

        divide = int(len(self.filtered_papers) * train_percent / 100)
        self.filtered_train_papers = self.filtered_papers[:divide]
        self.filtered_test_papers = self.filtered_papers[divide:] if divide < len(self._papers) else []

        self.loop_train = True  # whether __iter__ should loop over train or test papers.

    def toggle_loop_train(self):

        """Whether to loop over the train abstracts or the test abstracts.
        This influences any iteration over instances."""

        self.loop_train = False if self.loop_train else True

    def activate_loop_train(self):

        self.loop_train = True

    def activate_loop_test(self):

        self.loop_train = False

    def apply_filter(self, f):
        self.filter_by.append(f)

    def remove_all_filters(self):
        self.filter_by = list()

    def remove_filter(self, f):
        self.filter_by.remove(f)

    def __iter__(self):
        looped_over = self.filtered_train_papers if self.loop_train else self.filtered_test_papers
        for paper in looped_over:
            if not self.filter_by or (paper["classification"] and paper["classification"] in self.filter_by):
                try:
                    abstract = paper["abstract"]
                    if abstract:
                        yield self.parse_line(abstract)
                    else:
                        print("Abstract empty for paper {}".format(paper["id"]))
                        yield []

                except KeyError:
                    print("no abstract for paper {}".format(paper["id"]))
        raise StopIteration

    def __getitem__(self, index):  # ignores filters
        if self.filter_by:
            print("WARNING: __getitem__ for class PaperReader is ignoring filters {}".format(self.filter_by))
        abstract = self._papers[index]["abstract"]
        if abstract:
            return self.parse_line(abstract)
        else:
            return None

    def __len__(self):
        return len(self._papers)

    def parse_line(self, line):
            line = line.lower()
            words = line.split()
            clean = [self.parse_word(word) for word in words]
            return list(filter(lambda l: l, clean))

    def parse_word(self, word):
        for ch in self.dismiss:
            if ch in word:
                return None
        return "".join(ch for ch in word if ch in self.keep)

    def generate_words_list(self):
        print("Generating list of words from abstracts with filters {}".format(self.filter_by))
        i = 0
        self.words = []
        for abstract in self:
            if abstract:
                self.words += abstract
            if i % 50000 == 0 and i > 0:
                print("{} papers parsed so far".format(i))
            i += 1
        print("Total word count: {}\nTotal papers: {}".format(len(self.words), i))

    def save_words(self, binary=False):
        if binary:
            with open("words", "wb") as w:
                pickle.dump(self.words, w)
        else:
            with open("words.txt", 'w') as words_file:
                for w in self.words:
                    words_file.write("{}\n".format(w))

    def save_train_papers(self, path):

        """Pickle dump the train papers."""

        with open(path, "wb") as trs:
            pickle.dump(self.filtered_train_papers, trs)

    def save_test_papers(self, path):

        """Pickle dump the test papers."""

        with open(path, "wb") as tes:
            pickle.dump(self.filtered_test_papers, tes)

    def dump_text(self, path):
        with open(path, 'w') as text:
            for word in self.words:
                text.write('{} '.format(word))

    def generate_train_text_file(self, path):

        """Generate a text file with every word from the training abstracts"""

        self.generate_words_list()
        self.dump_text(path)


if __name__ == '__main__':

    with open("documents_array.json", encoding="utf-8") as docs:
        reader = PaperReader(json.load(docs), filters=["systematic-review", "primary-study"], train_percent=80,
                             abstracts_min=30)
    reader.generate_words_list()
    reader.dump_text("episte_sys_prima")