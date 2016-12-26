import string
import pickle
import json


class PaperReader:

    def __init__(self, papers, filters=None, train_percent=100, abstracts_min_lmt=None, dispose_no_abstract=True):

        self.abstracts_min_lmt = abstracts_min_lmt  # used to dispose short abstracts and
        # to limit the words used to generate words' list

        # Dispose of short (if abstracts_min is True) or empty abstracts otherwise
        self._papers = []
        for paper in papers:
            if abstracts_min_lmt and paper["abstract"] and len(paper["abstract"].split()) >= abstracts_min_lmt:
                self._papers.append(paper)
            elif not abstracts_min_lmt and paper["abstract"] and dispose_no_abstract:
                self._papers.append(paper)
            elif not abstracts_min_lmt and not dispose_no_abstract:
                self._papers = papers
                break

        divide = int(len(self._papers) * train_percent / 100)
        self._train_papers = self._papers[:divide]
        self._test_papers = self._papers[divide:] if divide < len(self._papers) else self._papers

        self.keep = string.ascii_lowercase + '-'
        self.dismiss = "123456789"
        self.words = []
        self.filter_by = filters if filters else []  # filter by paper classification (systematic-review,
        # structured-summary-of-systematic-review, primary-study, overview, structured-summary-of-primary-study)

        self.loop_train = True  # whether __iter__ should loop over train or test papers.

    def toggle_loop_train(self):
        self.loop_train = False if self.loop_train else True

    def apply_filter(self, f):
        self.filter_by.append(f)

    def remove_all_filters(self):
        self.filter_by = list()

    def remove_filter(self, f):
        self.filter_by.remove(f)

    def __iter__(self):
        looped_over = self._train_papers if self.loop_train else self._test_papers
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
                self.words += abstract[:self.abstracts_min_lmt] if self.abstracts_min_lmt else abstract
            if i % 50000 == 0 and i > 0:
                print("{} papers parsed so far".format(i))
            i += 1
        print("Total word count: {}\nTotal papers: {}".format(len(self.words), i))

    def get_filtered_train_papers(self):
        return [p for p in filter(lambda paper: paper["classification"] in self.filter_by if self.filter_by else True,
                                  self._train_papers)]

    def get_filtered_test_papers(self):
        return [p for p in filter(lambda paper: paper["classification"] in self.filter_by if self.filter_by else True,
                                  self._test_papers)]

    def save_words(self, binary=False):
        if binary:
            with open("words", "wb") as w:
                pickle.dump(self.words, w)
        else:
            with open("words.txt", 'w') as words_file:
                for w in self.words:
                    words_file.write("{}\n".format(w))

    def save_train(self, path):
        with open(path, "wb") as trs:
            pickle.dump(self.get_filtered_train_papers(), trs)

    def save_test(self, path):
        with open(path, "wb") as tes:
            pickle.dump(self.get_filtered_test_papers(), tes)

    def dump_text(self, path):
        with open(path, 'w') as text:
            for word in self.words:
                text.write('{} '.format(word))

if __name__ == '__main__':

    with open("documents_array.json", encoding="utf-8") as docs:
        reader = PaperReader(json.load(docs), filters=["systematic-review", "primary-study"], train_percent=80,
                             abstracts_min_lmt=30)
    reader.generate_words_list()
    reader.dump_text("episte_sys_prima")