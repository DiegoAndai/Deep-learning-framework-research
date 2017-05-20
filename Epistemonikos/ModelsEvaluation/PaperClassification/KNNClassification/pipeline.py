import classify
import json

k_to_test = [10, 50, 100, 500, 5000, 10000, False]
results = {10: [], 50: [], 500: [], 5000: [], 10000: [], False: []}

for k in [100]:
    for i in range(5):
        acc, conf, restricted_dict = classify.main(k, i, restrict_random = True)
        conf_list = [list(conf[0]), list(conf[1])]
        results[k].append([acc, conf_list, k, restricted_dict])
    with open("results_denis_new_round_{}k.json".format(k), "w") as json_out:
        json.dump(results, json_out)

    print("test with k: {} completed".format(k))
