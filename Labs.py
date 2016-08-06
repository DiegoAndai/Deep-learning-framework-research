from stackexchange import *
from time import sleep


class ResultSetLab:

    """Super analyzer."""

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


class TagLab(ResultSetLab):

    def __init__(self):
        super().__init__()

    def get_related(self, tag_name):
        self.result_set = self.site.build('tags/%s/related' % tag_name, Tag, 'tag')

    def get_tags(self):
        self.result_set = self.site.build('tags/', Tag, 'tag')

    def get_tag_synonyms(self, tag_name):
        self.result_set = self.site.build('tags/{}/synonyms'.format(tag_name), Tag, 'tag')


if __name__ == '__main__':
    lab = TagLab()
    lab.get_related("caffe")
    for tag in lab.result_set:
        print(tag.name)
    #lab.get_all_items()
