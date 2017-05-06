import classify
import json

k_to_test = [5, 10, 20, 50, 500, False]
results = {5: [], 10: [], 20: [], 50: [], 500: [], False: []}

for k in k_to_test:
    for _ in range(5):
        acc, conf, restricted_dict = classify.main(k, restrict_random = True)
        conf_list = [list(conf[0]), list(conf[1])]
        results[k].append([acc, conf_list, k, restricted_dict])
        with open("results_no_exclusives_random_restrict.json", "w") as json_out:
            json.dump(results, json_out)

    print("test with k: {} completed".format(k))
