import pickle
import numpy as np
import matplotlib.pyplot as plt

def color(ordered):
    if ordered:
        return 'b'
    return 'r'


def filter_dict(input_dict, classified, classification):
    usable = list(filter(lambda r: r["meta"]["classificated"] == classified, input_dict.values()))
    usable = list(filter(lambda r: r["meta"]["classification"] == classification, usable))
    return usable

def mine_words_data(input_dict, classified, classification):
    usable = filter_dict(input_dict, classified, classification)
    word_occurrences = [results["words_occurrence"] for results in usable]
    words = dict()
    for occurrences_table in word_occurrences:
        for word, value in occurrences_table.items():
            if word in words:
                words[word] += value
            else:
                words[word] = value
    return words

def top_from_word_table(table, qty, max_option = True):
    if qty == "all":
        qty = len(table)
    top = sorted(table.items(), key = lambda t: t[1], reverse = max_option)
    top_n = top[:qty]
    return top_n

with open("Max_pool_lab_results8", "rb") as order_file, \
     open("Max_pool_lab_results9", "rb") as shuffled_file, \
     open("Max_pool_lab_results10", "rb") as random_file:

    order_results = pickle.load(order_file)
    shuffled_results = pickle.load(shuffled_file)
    random_results = pickle.load(random_file)




result_dict = {"ordered": order_results,
               "shuffled": shuffled_results,
               "random": random_results}

classes = ["systematic-review", "primary-study"]
classified = ["correctly", "wrong"]

n = 10
poly_n = 2
text_log = ""

'''fig, axes = plt.subplots(nrows=2, ncols=3)
ax0, ax1, ax2, ax3, ax4, ax5 = axes.flatten()

fig_dict = {"ordered_systematic-review": ax0,
            "shuffled_systematic-review": ax1,
            "random_systematic-review": ax2,
            "ordered_primary-study": ax3,
            "shuffled_primary-study": ax4,
            "random_primary-study": ax5}'''


for _type in ["ordered", "shuffled", "random"]:
    words = dict()
    for _classified in classified:
        for _class in classes:
            results = mine_words_data(result_dict[_type], _classified, _class)
            total_occurrences = sum((result[1] for result in results.items()))
            output = "Top {} word occurrences in {} when {} and classified {}:\n".format(n, _class, _type, _classified)
            for word_tuple in top_from_word_table(results, 100):
                word = word_tuple[0]
                qty = word_tuple[1]
                output +=   "-{}: {}\n".format(word, qty/total_occurrences)
                if word in words:
                    words[word][_class] = qty/total_occurrences
                else:
                    words[word] = {_class: qty/total_occurrences}
            text_log += "\n{}\n".format(output)
        color = ("b" if _classified == "correctly" else "r")
        sr = []
        ps = []
        for word, values in words.items():
            if "systematic-review" in values:
                sr.append(values["systematic-review"])
            else:
                sr.append(0)
            if "primary-study" in values:
                ps.append(values["primary-study"])
            else:
                ps.append(0)
        poly = np.poly1d(np.polyfit(sr, ps, poly_n))
        xp = np.linspace(min(sr), max(sr), 100)
        integrand = poly.integ()
        print(np.arctan(poly[1]))
        plt.xlim(min(sr) - 0.0002, max(sr) + 0.0002)
        plt.ylim(min(ps) - 0.0002, max(ps) + 0.0002)
        plt.plot(sr, ps, 'o', xp, poly(xp), '--', color = color)
    #plt.savefig("{}_graph".format(_type))
    plt.show()

with open("top_words.txt","w") as text_log_output:
    text_log_output.write(text_log)


'''fig, axes = plt.subplots(nrows=2, ncols=2)
ax0, ax1, ax2, ax3 = axes.flatten()

scatter_dict = {"systematic-review_correctly": ax0,
                "primary-study_correctly": ax1,
                "systematic-review_wrong": ax2,
                "primary-study_wrong": ax3}
class_dict_word_appareance = dict()


for _class in classes:
    for _classified in classified:
        for _type in ["shuffled", "ordered"]:
            word_dict = {} #contains a list with the occurrences for certain word in every document, for later probability study
            document_appareance_dict = {}
            on_this_category = filter_dict(result_dict[_type], _classified, _class)
            for document in on_this_category:
                word_table = document["words_occurrence"]
                for word, qty in word_table.items():
                    if word in word_dict:
                        word_dict[word].append(qty)
                        document_appareance_dict[word] += 1
                    else:
                        word_dict[word] = [qty]
                        document_appareance_dict[word] = 1
            prom_table = {key: sum(value)/len(value) for key, value in word_dict.items()}
            print(max_from_word_table(prom_table, n))

            x = list()
            y = list()

            for word, prom in prom_table.items():
                document_appareance = document_appareance_dict[word]
                y.append(prom)
                x.append(document_appareance)

            scatter_dict["{}_{}".format(_class, _classified)].scatter(x,y, color = color(_type == "ordered"))
plt.show()'''
