import argparse
import pickle
from PVPClassifier import PVPClassifier
from tabulate import tabulate


parser = argparse.ArgumentParser(description="Classify papers.")
parser.add_argument("--classes", nargs='+', help="Paper types to classify to.", required=True)
parser.add_argument("--model_path", help="Path to a file containing pickled embeddings as a numpy array.",
                    required=True)
parser.add_argument("--words_path", help="Path to a pickled list of the words in the model, "
                                         "in the order of the embeddings.",
                    required=True)
parser.add_argument("--ref_papers_path", help="Path to pickled list of papers (a paper should be represented as a "
                                              "dictionary with at least abstract and a classification keys containing "
                                              "the abstract and the paper type) to generate vectors that will be used"
                                              "as reference to classify papers.",
                    required=True)
parser.add_argument("--papers_path", help="Path to pickled list of papers (see ref_papers_path help) to classify.",
                    required=True)
parser.add_argument("--abstracts_words", type=int, default=10,
                    help="Words from the abstracts to consider to build vectors.")
parser.add_argument("--save_into", help="Path to file to save classification output.", default='')  # not yet used
args = parser.parse_args()


with open(args.model_path, 'rb') as model_file, \
     open(args.words_path, 'rb') as words_file, \
     open(args.ref_papers_path, 'rb') as ref_file, \
     open(args.papers_path, 'rb') as abs_file:

    model = pickle.load(model_file)
    model_order = pickle.load(words_file)
    ref_papers = pickle.load(ref_file)
    to_classify = pickle.load(abs_file)[:1000]

classifier = PVPClassifier(model, model_order, args.classes, ref_papers, span=args.abstracts_words)
classifier.get_ref_vectors(new_n_save=True)
classifier.get_abs_vectors(to_classify, new_n_save=True)
classifier.classify()

print(classifier.get_conf_mat_pretty())
print("Accuracy:", classifier.get_accuracy())
classifier.print_recall()
classifier.print_precision()
