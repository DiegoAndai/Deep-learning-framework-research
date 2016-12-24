import string
import pickle


class PaperReader:

    def __init__(self, papers, filters=None, balance_train=False,
                 train_percent=100, abstracts_min=None, dispose_no_abstract=True):

        self.abstracts_min = abstracts_min  # used to dispose short abstracts and
        # to limit the words used to generate words' list

        # Dispose of short (if abstracts_min is True) or empty abstracts otherwise
        self.papers = []
        for paper in papers:
            if abstracts_min and len(paper["abstract"]) >= abstracts_min:
                self.papers.append(paper)
            elif not abstracts_min and paper["abstract"] and dispose_no_abstract:
                self.papers.append(paper)
            elif not abstracts_min and not dispose_no_abstract:
                self.papers = papers
                break

        divide = int(len(self.papers) * train_percent / 100)
        self.train_papers = self.papers[:divide]
        self.test_papers = self.papers[divide:] if divide < len(self.papers) else self.papers

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
        looped_over = self.train_papers if self.loop_train else self.test_papers
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
        abstract = self.papers[index]["abstract"]
        if abstract:
            return self.parse_line(abstract)
        else:
            return None

    def __len__(self):
        return len(self.papers)

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
                self.words += abstract[:self.abstracts_min] if self.abstracts_min else abstract
            if i % 50000 == 0 and i > 0:
                print("{} papers parsed so far".format(i))
            i += 1
        print("Total word count: {}\nTotal papers: {}".format(len(self.words), i))

    def get_train_papers(self):
        return self.train_papers

    def get_test_papers(self):
        return self.test_papers

    def save_words(self, binary=False):
        if binary:
            with open("words", "wb") as w:
                pickle.dump(self.words, w)
        else:
            with open("words.txt", 'w') as words_file:
                for w in self.words:
                    words_file.write("{}\n".format(w))
