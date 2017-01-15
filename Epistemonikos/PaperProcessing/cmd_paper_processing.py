from argparse import ArgumentParser
from open_documents import PaperReader
from os.path import join
import json


def dump_runtime_info(args_namespace, file_name, paper_reader):

    with open(join(args_namespace.save_path, file_name), 'w', encoding="utf-8") as info_file:
        info_file.write("cmd_paper_processing.py options:\n")
        for param, value in vars(args_namespace).items():
            info_file.write("{}: {}\n".format(param, value))
        info_file.write("Train Papers: {}\nTest Papers: {}".format(len(paper_reader.filtered_train_papers),
                                                                   len(paper_reader.filtered_test_papers)))

parser = ArgumentParser()
parser.add_argument("--json_file", help="Path to json file with papers from which a set will be built.",
                    required=True)
parser.add_argument("--filters", default=None, nargs='+',
                    help="Filter papers by their type. Only consider the types given. If omitted, "
                         "every type is considered.")
parser.add_argument("--train_percent", default=100, type=float,
                    help="Percentage of the filtered papers to take for training. "
                         "Rest will be considered for test set.")
parser.add_argument("--min_count", default=None, type=int,
                    help="Minimum count of words each abstract must have. Any paper with"
                         "less words will not be disposed of. If omitted, every abstract"
                         "will be considered.")
parser.add_argument("--dispose_empty", default=True, help="Whether to dispose of the empty abstracts or not.")
parser.add_argument("--even_train", default=False)
parser.add_argument("--even_test", default=False)
parser.add_argument("--mode", default=0, type=int, help="0: Generate text file to train a language model."
                                                        "1: Generate a PVP classification set."
                                                        "2: Generate a KNN classification set."
                                                        "3: All of the preceding.")
parser.add_argument("--save_path", default="", help="Where to save the output file(s).")
parser.add_argument("--years", default=None, help="Filter by years", nargs='+')
args = parser.parse_args()


with open(args.json_file, encoding="utf-8") as jf:
    papers = json.load(jf)

paper_reader = PaperReader(papers, args.filters, args.train_percent, args.min_count,
                           args.dispose_empty, args.even_train, args.even_test, args.years)

m = args.mode
if m == 0 or m == 3:

    paper_reader.generate_train_text_file(join(args.save_path, "train_text.txt"))

if m == 1 or m == 3:

    paper_reader.save_train_papers(join(args.save_path, "train_papers"))
    paper_reader.save_test_papers(join(args.save_path, "test_papers"))

if m == 2 or m == 3:

    paper_reader.save_train_papers(join(args.save_path, "train_papers"), form="json")
    paper_reader.save_test_papers(join(args.save_path, "test_papers"), form="json")


dump_runtime_info(args, "creation_info.txt", paper_reader)
