from Labs import *
import matplotlib.pyplot as plt
from random import randint

r = lambda: randint(0, 255)


class OccurrencePlot:

    """Class to plot data from labs according to creation date."""

    def __init__(self, result_set_lab):

        """parameter result_set_lab: ResultSetLab object"""

        self.lab = result_set_lab
        self.result_sets_occurrence = {}  # occurrences stored as result_set_id:occurrence_list/creation_dates
        self.color_map = []

        """Styling"""

        ax = plt.subplot(111)
        ax.spines["top"].set_visible(False)
        ax.spines["bottom"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["left"].set_visible(False)
        ax.get_xaxis().tick_bottom()
        ax.get_yaxis().tick_left()

    def add_result_set(self, result_set_name):

        """This method assumes the result set with name result_set_name in the lab already has items that have a
        creation date"""

        self.result_sets_occurrence[result_set_name] = self.lab.get_creation_dates(result_set_name)
        self.color_map.append('#{:02x}{:02x}{:02x}'.format(r(), r(), r()))

    def rm_result_set(self, result_set_name):
        self.result_sets_occurrence.pop(result_set_name, None)
        self.color_map.pop()

    @staticmethod
    def set_date_domain(year_start, month_start, year_finish, month_finish):
        if datetime.datetime(year_finish,month_finish,31,0,0,0)>datetime.datetime.today():
            plt.xlim(datetime.datetime(year_start,month_start,1,0,0,0), datetime.datetime.today())
        else:
            plt.xlim(datetime.datetime(year_start,month_start,1,0,0,0), datetime.datetime(year_finish,month_finish,31,0,0,0))

    def plot_by_month(self):
        color_index = 0
        """plots time count for every added result_set"""
        for (result_set_name, occurrence_list) in self.result_sets_occurrence.items():

            """Group occurrences by year and month"""

            occurrence_date_list = [datetime.datetime(item.year,item.month,1,0,0,0) for item in occurrence_list]

            """Create month list so no month gets left out even if there's no entry for it"""
            actual_date=occurrence_date_list[0]
            month_list=[actual_date]
            while actual_date!=occurrence_date_list[-1]:
                if actual_date.month==12:
                    actual_year=actual_date.year+1
                    actual_month=1
                    actual_date=datetime.datetime(actual_year,actual_month,1,0,0,0)
                else:
                    actual_month=actual_date.month+1
                    actual_date=datetime.datetime(actual_date.year,actual_month,1,0,0,0)
                month_list.append(actual_date)

            """Create a dict with datetime format date and number of occurrences"""

            occurrence_count_list = [occurrence_date_list.count(month) for month in month_list]
            plt.plot(month_list, occurrence_count_list, lw=2.5,
                     label=('{}: {}'.format(result_set_name, sum(occurrence_count_list))),
                     color=self.color_map[color_index])
            color_index += 1
        plt.legend(loc=2)
        plt.savefig('Framework popularity by question count per month in 2016')
        plt.show()
        plt.gcf().clear()


if __name__ == '__main__':

    #Example usage:
#    Lab=QuestionLab('Tags') #creates the ResultLab object with a 'Tags' result_set_name
#    Plot=OccurrencePlot(Lab) #creates ocurrence plot object, with its respective lab
#    Plot.lab.get_questions('mxnet',result_set_name='Tags') #gets all questions with the desired tan ('mxnet' here) in the result_set_name ('Tags')
#    Plot.add_result_set('Tags') #adds the result set, in the format: {result_set_id:ocurrence_list} to the plot object
#    Plot.plot_by_month()


    fmwks=['mxnet','keras','tensorflow','caffe','torch','deeplearning4j','theano','lasagne','pybrain']
    Lab=QuestionLab()
    Plot=OccurrencePlot(Lab)
    for fmwk in fmwks:
        rs_name='Tagged{}'.format(fmwk.title())
        Plot.lab.add_result_set(rs_name)
        Plot.lab.get_questions(fmwk,result_set_name=rs_name)
        Plot.add_result_set(rs_name)
    Plot.set_date_domain(2016,1,2016,12)
    Plot.plot_by_month()
