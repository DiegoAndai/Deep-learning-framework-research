from stackexchange import *
from time import sleep


class ResultSetLab:

    """Analyzer for questions."""

    def __init__(self, result_set=None, site=StackOverflow):
        self.result_set = result_set
        self.site = Site(site)
        self.item = "item"

    def get_all_items(self):
        while self.result_set.has_more:
            try:
                print("Current {} count: {}. Getting more items...".format(self.item, len(self.result_set)))
                self.result_set = self.result_set.extend_next()
            except StackExchangeError:
                sleep(5)
        print("Final {} count: {}".format(self.item, len(self.result_set)))


class QuestionLab(ResultSetLab):

    """Question Analyzer"""

    def __init__(self, *tags, pagesize=100):
        super().__init__()
        self.result_set = self.site.questions(tagged=[tag for tag in tags], pagesize=pagesize)
        self.item = "question"


if __name__ == '__main__':
    lab = ResultSetLab()
    lab.result_set = lab.site.questions(pagesize=100, tagged=["chrono"])
    lab.get_all_items()
