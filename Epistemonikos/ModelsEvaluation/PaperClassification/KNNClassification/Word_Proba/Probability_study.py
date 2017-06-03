import json
import matplotlib.pyplot as plt
from numpy import log

with open("proba_results.json", "r") as json_in:
    data = json.load(json_in)

denominator = "ps_proba"
numerator = "sr_proba"

proba_ratios = {}
for word, probabilities in data.items():
    if probabilities[denominator] and probabilities[numerator]:
        proba_ratios[word] = probabilities[denominator] / probabilities[numerator]


results = sorted(proba_ratios.items(), key = lambda t: t[1], reverse = True)

plt.ylabel("{}/{}".format(denominator, numerator))
plt.xlabel("Orden de palabras con mayor radio")
for i in range(len(results)):
    plt.scatter(i, log(results[i][1]))

plt.savefig("proba_ratio_results_log.png")
#plt.show()

for word in results[:20]:
    print(word[0], data[word[0]][denominator])
print("\n")
for word in results[-20:]:
    print(word[0], data[word[0]][numerator])

with open("proba_ratio_results.json", "w") as json_out:
    json.dump(results, json_out)
