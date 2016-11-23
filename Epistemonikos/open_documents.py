import string


class PaperReader:

    def __init__(self, papers):
        self.papers = papers
        self.keep = string.ascii_lowercase + '-'
        self.dismiss = "123456789"
        self.words = []

    def __iter__(self):
        for paper in self.papers:
            try:
                abstract = paper["abstract"]
                if abstract:
                    yield self.parse_line(abstract)

            except KeyError:
                print("no abstract for paper {}".format(paper["id"]))
        raise StopIteration

    def __getitem__(self, index):
        abstract = self.papers[index]["abstract"]
        if abstract:
            return self.parse_line(abstract)
        else:
            return None

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
        i = 0
        for abstract in self:
            if abstract:
                self.words += abstract
            if i % 50000 == 0:
                print("{} papers parsed so far".format(i))
            i += 1
        print("Total word count:", len(self.words))

    def save_words(self):
        with open("words.txt", 'w') as words_file:
            for w in self.words:
                words_file.write("{}\n".format(w))
