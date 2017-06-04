import classify
import json

k_to_test = [80000]
results = {40000: [], 80000: [], 125000: []}

for k in k_to_test:
    for i in range(5):
        acc, conf, restricted_dict = classify.main(k, i, restrict_random = True)
        conf_list = [list(conf[0]), list(conf[1])]
        results[k].append([acc, conf_list, k])
    with open("results_denis_new_round_{}k.json".format(k), "w") as json_out:
        json.dump(results, json_out)

    print("test with k: {} completed".format(k))

'''python pipeline.py --K 10 --model_path ../../../LanguageModels/Skipgram/Advanced/TrainedModels/Model4 --span 80 --KNN_papers_set Set4
'''

#1
