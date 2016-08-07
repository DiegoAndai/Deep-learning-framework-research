from stackexchange import *
from time import sleep


class ResultSetLab:

    """Super analyser. Has a dict attribute to contain desired Py-StackExchange ResultSets."""

    def __init__(self, *result_set_ids, site=StackOverflow, item="item"):

        """:parameter result_set_ids: dict keys to identify result sets in self.result_sets."""

        self.result_sets = {typ: None for typ in result_set_ids}
        self.site = Site(site)
        self.item = item

    def get_all_items(self, result_set):

        """Extends specified result set with data until there is no more data to be retrieved."""

        result_set = self.result_sets[result_set]
        while result_set.has_more:
            try:
                print("Current {} count: {}. Getting more {}s...".format(self.item, len(result_set), self.item))
                result_set = result_set.extend_next()
            except StackExchangeError:
                sleep(5)
        print("Final {} count: {}".format(self.item, len(self.result_sets)))

    def get_all_for_all(self):

        """Calls get_all_items for each result set in self.result_set."""

        for name, rs in self.result_sets.items():
            print("Retrieving {}s for result set {}...".format(self.item, name))
            self.get_all_items(rs)

    def show_rs_types(self):

        """Prints self.result_sets keys."""

        print(key for key in self.result_sets.keys())

    def add_result_set(self, key, value=None):

        """Appends a result set key (and optionally a value) to self.result_sets."""

        self.result_sets.update({key: value})

    def reset_rs(self):
        self.result_sets = {}


class QuestionLab(ResultSetLab):

    """Question analyser"""

    def __init__(self, *result_set_ids):
        super().__init__(*result_set_ids, item="question")

    def get_questions(self, *tags, result_set, pagesize=100):

        """Retrieves pagesize questions into specified result set."""

        self.result_sets[result_set] = self.site.questions(tagged=[tag for tag in tags], pagesize=pagesize)


class TagLab(ResultSetLab):

    """Tag analyser"""

    def __init__(self, *result_set_ids):
        super().__init__(*result_set_ids, item="tag")

    def get_related(self, tag_name, result_set):
        self.result_sets[result_set] = self.site.build('tags/{}/related'.format(tag_name), Tag, 'tag')

    def get_tags(self, result_set):
        self.result_sets[result_set] = self.site.build('tags/', Tag, 'tag')

    def get_tag_synonyms(self, tag_name, result_set_name):
        self.result_sets[result_set_name] = self.site.build('tags/{}/synonyms'.format(tag_name), TagSynonym, 'tag')


if __name__ == '__main__':

    # Example usage:
    lab = TagLab("All", "PyRel")  # Create a tag lab with two keys for result sets in self.result_sets: All and PyRel.
    lab.get_related("python", "PyRel")  # initialize a result set in the "PyRel" key with at most 100 tags related to
    # python
    lab.get_tags("All")  # initialize a rs in the "All" key with at most 100 tags.
    lab.add_result_set("PySyn")
    lab.get_tag_synonyms("python", "PySyn")
    for tag in lab.result_sets["PySyn"]:
        print(tag.from_tag)
