import pickle

with open("Max_pool_lab_results1", "rb") as one_file, \
     open("Max_pool_lab_results2", "rb") as two_file:

    results1 = pickle.load(one_file)
    results2 = pickle.load(two_file)

for _id, results in results1.items():
    shuffled_results = results2[_id]

    indexes1 = results["indexes_occurrence"]
    indexes2 = shuffled_results["indexes_occurrence"]
    difference_bar = "|"
    for i in range(0, 80):
        if i in indexes1:
            value1 = indexes1[i]
        else:
            value1 = 0
        if i in indexes2:
            value2 = indexes2[i]
        else:
            value1 = 0
        difference = value1 - value2
        difference_bar += "{}|".format(difference)
    print(difference_bar)
