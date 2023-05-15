import argparse

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "-r",
        "--reference-file",
        type=str,
        default='/home/sli136/l2mt/data/nmt-grammar-noise/en-es-experiments/train.clean.en',
        help="source file prefix"
    )
    parser.add_argument(
        "-t",
        "--target-file",
        type=str,
        default='/home/sli136/l2mt/data/nmt-grammar-noise/en-es-experiments/train.prep.en',
        help="target output prefix"
    )
    return parser.parse_args()

args = parse_args()
for arg in vars(args):
    print(arg, getattr(args, arg))
    
num_diff = 0
total_sent = 0
with open(args.reference_file, "r") as f_en:
    with open(args.target_file, "r") as f_es:
        for line_i, (line_en, line_es) in enumerate(zip(f_en.readlines(), f_es.readlines())):
            total_sent += 1
            if " ".join(line_en.strip().split()) != " ".join(line_es.strip().split()):
                num_diff += 1

print("Done counting, {} sentences are different out of {} sentences ({}%)".format(num_diff, total_sent, num_diff/total_sent*100))