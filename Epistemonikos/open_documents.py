import string

class SentenceReader:

    def __init__(self, files):
        self.files = files
        self.keep = string.ascii_lowercase + '-'
        self.dismiss = "123456789"

    def __iter__(self):
        for file in self.files:
            try:
                abstract = file["abstract"]
                if abstract:
                    yield self.parse_line(abstract)

            except KeyError:
                print("no abstract for file {}".format(file["id"]))
        raise StopIteration

    def __getitem__(self, index):
        abstract = self.files[index]["abstract"]
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
