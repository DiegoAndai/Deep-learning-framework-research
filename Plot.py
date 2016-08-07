from Labs import *
import matplotlib.pyplot as plt
from random import randint

r= lambda: randint(0,255)

class OccurrencePlot():

    def __init__(self, result_set_lab):

        """parameter result_set_lab: ResultSetLab object"""

        self.lab=result_set_lab

        """occurrences stored as result_set_id:ocurrence_list key:value pair"""

        self.result_sets_ocurrence={}
        self.color_map=[]

    def add_result_set(self, result_set_name):

        """This method assumes the result_set_name already has data with items that have a creation date"""

        self.result_sets_ocurrence[result_set_name]=self.lab.get_occurrence_list(result_set_name)
        self.color_map.append('#{:02x}{:02x}{:02x}'.format(r(),r(),r()))

    def rm_result_set(self, result_set_name):
        self.result_sets_ocurrence.pop(result_set_name, None)
        self.color_map.pop()

    def plot_by_month(self):

        #for result_set_name in self.result_sets:

        """Group occurrences by year and month"""

        self.occurrence_year_month_list=[(item.creation_date.year,item.creation_date.month) for item in self.result_sets[result_set_name]]

        """Create a dict with datetime format date and number of occurrences"""

        self.occurrence_count_dict={datetime(year,month,1,0,0,0):self.occurrence_year_month_list.count((year,month)) for (year,month) in  self.occurrence_year_month_list}


if __name__ == '__main__':

    #Example usage:
    Lab=TagLab('Tags') #creates the ResultLab object with a 'Tags' result_set_name
    Plot=OccurrencePlot(Lab) #creates ocurrence plot object, with its respective lab
    Plot.lab.get_tagged_questions('mxnet','Tags') #gets all questions with the desired tan ('mxnet' here) in the result_set_name ('Tags')
    Plot.add_result_set('Tags') #adds the result set, in the format: {result_set_id:ocurrence_list} to the plot object
    print(Plot.result_sets_ocurrence)
