import classify
import json

k_to_test = [40000, 80000, 125000]
results = {40000: [], 80000: [], 125000: []}

for k in [100]:
    for i in range(5):
        acc, conf, restricted_dict = classify.main(k, i, restrict_random = True)
        conf_list = [list(conf[0]), list(conf[1])]
        results[k].append([acc, conf_list, k, restricted_dict])
    with open("results_denis_new_round_{}k.json".format(k), "w") as json_out:
        json.dump(results, json_out)

    print("test with k: {} completed".format(k))
