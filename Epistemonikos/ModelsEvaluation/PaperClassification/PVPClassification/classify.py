import argparse
import pickle
import json
import sys
from CentroidClassifier import CentroidClassifier

# TODO: reduce the amount of command line arguments assuming that the files related to one model are in the same path
parser = argparse.ArgumentParser(description="Classify papers.")
parser.add_argument("--classes", nargs='+', help="Paper types to classify to.", required=True)
parser.add_argument("--model_path", help="Path to a file containing pickled embeddings as a numpy array.",
                    required=True)
parser.add_argument("--path_to_vocab", help="Path to the vocab file of the model (as created by word2vec_optimized).",
                    required=True)
parser.add_argument("--ref_papers_path", help="Path to json file with papers (a paper should be represented as a "
                                              "dictionary with at least abstract and a classification keys containing "
                                              "the abstract and the paper type) to generate vectors that will be used"
                                              "as reference to classify papers.",
                    required=True)
parser.add_argument("--papers_path", help="Path to json file with papers (see ref_papers_path help) to classify.",
                    required=True)
parser.add_argument("--abstracts_words", type=int, default=10,
                    help="Words from the abstracts to consider to build vectors.")
parser.add_argument("--save_into", help="Path to file to save classification output. Output will be printed to stout "
                                        "nonetheless", default='')
args = parser.parse_args()


with open(args.model_path, 'rb') as model_file, \
     open(args.path_to_vocab, 'r') as words_file, \
     open(args.ref_papers_path, 'r') as ref_file, \
     open(args.papers_path, 'r') as abs_file:

    model = pickle.load(model_file)
    model_order = [line.split()[0].strip("b'") for line in words_file]
    ref_papers = json.load(ref_file)
    to_classify = json.load(abs_file)

classifier = CentroidClassifier(model, model_order, args.classes, ref_papers, span=args.abstracts_words)
classifier.get_ref_vectors(new_n_save=True)
classifier.get_abs_vectors(to_classify, new_n_save=True)
classifier.classify()


# print to stdout
def print_output(out=sys.stdout):

    cmat = classifier.get_conf_mat_pretty()
    acc = classifier.get_accuracy()
    print(cmat, end="\n\n", file=out)
    print("Accuracy:", acc, end="\n\n", file=out)

    print("Recall:", file=out)
    for cls, rcl in classifier.recalls():
        print(cls, rcl, sep=': ', file=out)

    print("\nPrecision:", file=out)
    for cls, prec in classifier.precisions():
        print(cls, prec, sep=': ', file=out)

    print(file=out)
    print(args.abstracts_words, "words from the abstracts were used to classify.", file=out)
    print(len(to_classify), "papers were classified.", file=out)

print_output()

if args.save_into:
    with open(args.save_into, 'w') as o:
        print_output(o)

