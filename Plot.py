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
        color_index=0
        """plots time count for every added result_set"""
        for (result_set_name, occurrence_list) in self.result_sets_ocurrence.items():

            """Group occurrences by year and month"""

            self.occurrence_date_list=[datetime.datetime(item.year,item.month,1,0,0,0) for item in occurrence_list]
            self.month_set=set(self.occurrence_date_list)
            self.month_list=(sorted(list(self.month_set)))


            """Create a dict with datetime format date and number of occurrences"""

            self.occurrence_count_list=[self.occurrence_date_list.count(month) for month in self.month_list]

            print(sum(self.occurrence_count_list))
            plt.plot(self.month_list, self.occurrence_count_list, 'bo-', label=result_set_name[6:] ,color=self.color_map[color_index])
            color_index+=1
        plt.legend(loc=2)
        plt.show()


if __name__ == '__main__':

    #Example usage:
#    Lab=TagLab('Tags') #creates the ResultLab object with a 'Tags' result_set_name
#    Plot=OccurrencePlot(Lab) #creates ocurrence plot object, with its respective lab
#    Plot.lab.get_tagged_questions('mxnet','Tags') #gets all questions with the desired tan ('mxnet' here) in the result_set_name ('Tags')
#    Plot.add_result_set('Tags') #adds the result set, in the format: {result_set_id:ocurrence_list} to the plot object
#    Plot.plot_by_month()

    fmwks=['mxnet','keras','torch','tensorflow','caffe']
    Lab=TagLab()
    Plot=OccurrencePlot(Lab)
    for fmwk in fmwks:
        rs_name='TagLab{}'.format(fmwk.title())
        Plot.lab.add_result_set(rs_name)
        Plot.lab.get_tagged_questions(fmwk,rs_name)
        Plot.add_result_set(rs_name)
    Plot.plot_by_month()
