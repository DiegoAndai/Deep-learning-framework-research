import pickle
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--path_to_vocab")
parser.add_argument("--path_to_words")
args = parser.parse_args()

with open(args.path_to_vocab) as vf, open(args.path_to_words, 'wb') as wf:
    pickle.dump([line.split()[0].strip("b'") for line in vf], wf)
