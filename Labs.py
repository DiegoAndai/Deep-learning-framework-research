from stackexchange import *
from time import sleep


class ResultSetLab:

    """Super analyser. Has a dict attribute to contain desired Py-StackExchange ResultSets."""

    def __init__(self, *result_set_ids, site=StackOverflow, item="item"):

        """:parameter result_set_ids: dict keys to identify result sets in self.result_sets."""

        self.result_sets = {typ: None for typ in result_set_ids}
        self.site = Site(site)
        self.item = item

    def get_all_items(self, result_set_name):

        """Extends specified result set with data until there is no more data to be retrieved."""

        result_set_name = self.result_sets[result_set_name]
        while result_set_name.has_more:
            try:
                print("Current {} count: {}. Getting more {}s...".format(self.item, len(result_set_name), self.item))
                result_set_name = result_set_name.extend_next()
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

    def get_occurrence_list(self, result_set_name):

        """Returns the list of creation dates for items in a result set, if they exist"""

        try:
            return [item.creation_date for item in self.result_sets[result_set_name]]
        except AttributeError:
            print("It appears that this result set doesn't have an associated creation time for each item.")

    def reset_rs(self):
        self.result_sets = {}


class QuestionLab(ResultSetLab):

    """Question analyser"""

    def __init__(self, *result_set_ids):
        super().__init__(*result_set_ids, item="question")

    def get_questions(self, *tags, result_set_name, pagesize=100):

        """Retrieves pagesize questions into specified result set."""

        self.result_sets[result_set_name] = self.site.questions(tagged=[tag for tag in tags], pagesize=pagesize)


class TagLab(ResultSetLab):

    """Tag analyser"""

    def __init__(self, *result_set_ids):
        super().__init__(*result_set_ids, item="tag")

    def get_tags(self, result_set_name=None, *tags):

        """Retrieves specified tags into specified result set and returns the latter."""

        tag_string = ""
        for tag in tags:
            tag_string = tag_string + tag + ";"
        self.result_sets[result_set_name] = self.site.build("tags/{}/info".format(tag_string), Tag, "tag")
        return self.result_sets[result_set_name]

    def get_related(self, tag_name, result_set_name):
        self.result_sets[result_set_name] = self.site.tag_related(tag_name)
        return self.result_sets[result_set_name]

    def get_all_tags(self, result_set_name):
        self.result_sets[result_set_name] = self.site.tags()
        return self.result_sets[result_set_name]

    def get_tag_synonyms(self, tag_name, result_set_name):
        self.result_sets[result_set_name] = self.site.build('tags/{}/synonyms'.format(tag_name), TagSynonym, 'tag')
        return self.result_sets[result_set_name]

    def get_tagged_questions(self, tag_name, result_set_name):
        self.result_sets[result_set_name] = self.site.search(tagged=tag_name)
        return self.result_sets[result_set_name]

if __name__ == '__main__':

    # Example usage:
    lab = TagLab("All", "PyRel")  # Create a tag lab with two keys for result sets in self.result_sets: All and PyRel.
    lab.get_related("python", "PyRel")  # initialize a result set in the "PyRel" key with at most 100 tags related to
    # python
    lab.get_all_tags("All")  # initialize a rs in the "All" key with at most 100 tags.
    lab.add_result_set("PySyn")
    lab.get_tag_synonyms("python", "PySyn")
    # for tag in lab.result_sets["PySyn"]:
    #     print(tag.from_tag)


    #Example of tags retrieved specifically by their names. They are automatically sorted
    for tag in lab.get_tags("ANN", "caffe", "openNN", "Keras", "Tensor Flow", "Mxnet", "Torch"):
        print(tag.name, tag.count)
