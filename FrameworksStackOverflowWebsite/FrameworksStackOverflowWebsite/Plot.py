from random import randint

import matplotlib.pyplot as plt

from FrameworksStackOverflowWebsite.FrameworksStackOverflowWebsite.Labs import *


def r():
    return randint(0, 255)


class OccurrencePlot:

    """Class to plot data from labs according to creation date."""

    def __init__(self, result_set_lab):

        """parameter result_set_lab: ResultSetLab object"""

        self.lab = result_set_lab
        self.result_sets_occurrences = {}  # occurrences stored as result_set_id:occurrence_list/creation_dates
        self.color_map = []

        """Styling"""

        ax = plt.subplot(111)
        ax.spines["top"].set_visible(False)
        ax.spines["bottom"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["left"].set_visible(False)
        ax.get_xaxis().tick_bottom()
        ax.get_yaxis().tick_left()

    def add_occurrences(self, result_set_name):

        """This method assumes the result set with name result_set_name in the lab already has items that have a
        creation date"""

        self.result_sets_occurrences[result_set_name] = self.lab.get_creation_dates(result_set_name)
        self.color_map.append('#{:02x}{:02x}{:02x}'.format(r(), r(), r()))

    def rm_result_set(self, result_set_name):
        self.result_sets_occurrences.pop(result_set_name, None)
        self.color_map.pop()

    @staticmethod
    def set_date_domain(start_date,finish_date):

        """dates are in datetime.datetime format"""

        if finish_date>datetime.datetime.today():
            plt.xlim(start_date, datetime.datetime.today())
        else:
            plt.xlim(start_date, finish_date)

    def plot_by_month(self):
        color_index = 0

        """plots time count for every added result_set"""

        for (result_set_name, creation_date_list) in self.result_sets_occurrences.items():

            """Group occurrences by year and month"""

            datetime_creation_dates_list = [datetime.datetime(creation_date.year,creation_date.month,1,0,0,0) for creation_date in creation_date_list]

            """Create month list so no month gets left out even if there's no entry for it"""
            current_date=datetime_creation_dates_list[0]
            month_list=[current_date]
            while current_date!=datetime_creation_dates_list[-1]:
                if current_date.month==12:
                    current_year=current_date.year+1
                    current_month=1
                    current_date=datetime.datetime(current_year,current_month,1,0,0,0)
                else:
                    current_month=current_date.month+1
                    current_date=datetime.datetime(current_date.year,current_month,1,0,0,0)
                month_list.append(current_date)

            """Create a dict with datetime format date and number of occurrences"""

            occurrence_count_list = [datetime_creation_dates_list.count(month) for month in month_list]
            plt.plot(month_list, occurrence_count_list, marker="o", lw=2.5,
                     label=('{}: {}'.format(result_set_name, sum(occurrence_count_list))),
                     color=self.color_map[color_index])
            color_index += 1

    def plot_by_week(self):
        color_index=0
        """plots time count for every added result_set"""
        for (result_set_name, occurrence_list) in self.result_sets_occurrences.items():

            """Group occurrences by year, month and week"""

            occurrence_date_list=[datetime.datetime(item.year,item.month,item.day,0,0,0) for item in occurrence_list]

            """Create week list so no week gets left out even if there's no entry for it"""

            delta=(occurrence_date_list[-1]-occurrence_date_list[0]).days
            delta_weeks=(delta)//7+1
            week_list=[occurrence_date_list[0]+datetime.timedelta(days=i*7) for i in range(delta_weeks+1)]

            week_occurrence_list=[]
            for occurrence in occurrence_date_list:
                for i in range(delta_weeks):
                    if occurrence>=week_list[i]:
                        try:
                            if occurrence<week_list[i+1]:
                                week_occurrence_list.append(week_list[i])
                        except IndexError:
                            week_occurrence_list.append(week_list[i])

            """Create a dict with datetime format date and number of occurrences"""

            occurrence_count_list=[week_occurrence_list.count(week) for week in week_list]
            plt.plot(week_list, occurrence_count_list, 'bo-', label=('{}: {}'.format(result_set_name,sum(occurrence_count_list))) ,color=self.color_map[color_index])
            color_index+=1

    def show_plot(self, legend=True):

        """shows all elements in the current plot"""
        if legend:
            plt.legend(loc=2)
        plt.show()

    def reset_plot(self):

        """empties plot"""
        plt.gcf().clear()

if __name__ == '__main__':

    #Example usage:
#    Lab=QuestionLab('Tags') #creates the ResultLab object with a 'Tags' result_set_name
#    Plot=OccurrencePlot(Lab) #creates ocurrence plot object, with its respective lab
#    Plot.lab.get_questions('mxnet',result_set_name='Tags') #gets all questions with the desired tan ('mxnet' here) in the result_set_name ('Tags')
#    Plot.add_occurrences('Tags') #adds the result set, in the format: {result_set_id:ocurrence_list} to the plot object
#    Plot.plot_by_month()


    fmwks=['keras','caffe','torch','tensorflow','keras','lasagne','mxnet']
    Lab=QuestionLab()
    Plot=OccurrencePlot(Lab)
    for fmwk in fmwks:
        rs_name='{}'.format(fmwk.title())
        Plot.lab.add_result_set(rs_name)
        Plot.lab.get_questions(fmwk, result_set_name=rs_name)
        Plot.add_occurrences(rs_name)

    Plot.set_date_domain(datetime.datetime(2016,1,1,0,0,0),datetime.datetime(2016,7,1,0,0,0))
    Plot.plot_by_month()
    plt.legend(loc=2)
    plt.savefig("./static/plots/overview_plot.png")
