import argparse
import shutil
from pathlib import Path
from itertools import islice
import sentencepiece as spm

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--src-file",
        nargs="*",
        type=str,
        default=["/home/sli136/l2mt/nmt-grammar-noise/en-es-experiments/train.clean.en"],
        help="source train file",
    )
    parser.add_argument(
        "--tgt-file",
        nargs="*",
        type=str,
        default=["/home/sli136/l2mt/nmt-grammar-noise/en-es-experiments/train.clean.es"],
        help="target train file"
    )
    parser.add_argument(
        "--encode-files",
        nargs="*",
        type=str,
        default=["/home/sli136/l2mt/nmt-grammar-noise/en-es-experiments/train.clean.es"],
        help="file(s) sources to encode using the trained/provided bpe model"
    )
    parser.add_argument(
        "--encode-dest",
        nargs="*",
        type=str,
        default=["/home/sli136/l2mt/nmt-grammar-noise/en-es-experiments/train.clean.es"],
        help="file(s) destinations to encode using the trained/provided bpe model"
    )
    parser.add_argument(
        "--model-dir",
        type=Path,
        default="/export/c11/sli136/l2mt/base_bpe",
        help="where the trained bpe model will be saved"
    )
    parser.add_argument(
        "--vocab-size",
        type=int,
        default=50000,
        help="Vocabulary size for BPE training"
    )
    parser.add_argument(
        "--train",
        action=argparse.BooleanOptionalAction,
        type=bool,
        default=False,
        help="train tokenizer"
    )
    parser.add_argument(
        "--encode",
        action=argparse.BooleanOptionalAction,
        type=bool,
        default=False,
        help="encode files"
    )
    parser.add_argument(
        "--transcript-words",
        type=str,
        default=None,
        help="one document containing all training sentences"
    )
    return parser.parse_args()


def train(args):
    vocab_size = args.vocab_size
    lang_dir = Path(args.model_dir)

    model_type = "bpe"
    model_prefix = f"{lang_dir}/{model_type}_{vocab_size}"

    if not args.transcript_words:
        train_text = []
        for filee in args.src_file:
            with open(filee, 'r') as f:
                train_text.extend([s.strip() for s in f.readlines()])
        for filee in args.tgt_file:
            with open(filee, 'r') as f:
                train_text.extend([s.strip() for s in f.readlines()])
        with open(f"{lang_dir}/transcript_words.txt", 'w') as f:
            f.write('\n'.join(train_text))
        train_text = f"{lang_dir}/transcript_words.txt"
    else:
        train_text = args.transcript_words

    character_coverage = 0.9995
    input_sentence_size = 100000000
    max_sentencepiece_length = 4

    # user_defined_symbols = ["[UNK]", "[CLS]", "[SEP]", "[PAD]", "[MASK]"]
    # unk_id = len(user_defined_symbols)
    # Note: unk_id is fixed to 2.
    # If you change it, you should also change other
    # places that are using it.

    model_file = Path(model_prefix + ".model")
    if not model_file.is_file():
        spm.SentencePieceTrainer.train(
            input=train_text,
            vocab_size=vocab_size,
            character_coverage=character_coverage,
            model_type=model_type,
            model_prefix=model_prefix,
            max_sentencepiece_length=max_sentencepiece_length,
            vocabulary_output_piece_score=False,
            input_sentence_size=input_sentence_size,
            # user_defined_symbols=user_defined_symbols,
            # unk_id=unk_id,
            bos_id=-1,
            eos_id=-1,
        )

    shutil.copyfile(model_file, f"{lang_dir}/bpe.model")

def encode(args):
    lang_dir = Path(args.model_dir)
    sp = spm.SentencePieceProcessor(model_file=str(f"{lang_dir}/bpe.model"))

    batch_size = 1000000

    for source, destination in zip(args.encode_files, args.encode_dest):
        with open(source, 'r') as f_in:
            print("================Processing file: {}================".format(source))
            for n_lines in iter(lambda: tuple(islice(f_in, batch_size)), ()):
                src_text = [s.strip() for s in n_lines]
                encoded_src = [' '.join(s) for s in sp.Encode(src_text, out_type=str)]
                print("encoded {} sentences to {}".format(len(encoded_src), destination))
                with open(destination, 'a+') as f_out:
                    f_out.writelines('\n'.join(encoded_src) + '\n')


if __name__ == "__main__":
    args = get_args()
    if args.train: train(args)
    if args.encode: encode(args)