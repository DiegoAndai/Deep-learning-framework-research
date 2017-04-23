import classify
import json

k_to_test = [5, 10, 20, 50, 100, 500, False]
results = []

for k in k_to_test:
    acc, conf = classify.main(k)
    results.append((acc, con, k))
    with open("results_no_exclusives.json", "w") as json_out:
        json.dump(results, json_out)

    print("test with k: {} completed".format(k))
