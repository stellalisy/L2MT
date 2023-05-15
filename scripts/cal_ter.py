import argparse
import os

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "-d",
        "--dir_name",
        type=str,
        default='./',
        help="results directory"
    )
    parser.add_argument(
        "-p",
        "--pred_file",
        type=str,
        default=None,
        help="source file"
    )
    parser.add_argument(
        "-t",
        "--tgt_file",
        type=str,
        default=None,
        help="target file"
    )
    parser.add_argument(
        "--ter",
        action='store_true',
        help="use ter"
    )
    parser.add_argument(
        "--sacrebleu",
        action='store_true',
        help="use sacrebleu"
    )
    parser.add_argument(
        "--nltk",
        action='store_true',
        help="use nltk bleu"
    )
    parser.add_argument(
        "--bleu",
        action='store_true',
        help="use bleu"
    )
    parser.add_argument(
        "--bertscore",
        action='store_true',
        help="use bertscore"
    )
    parser.add_argument(
        "--meteor",
        action='store_true',
        help="use meteor"
    )
    parser.add_argument(
        "--spm",
        type=str,
        default=None,
        help="spm model"
    )
    return parser.parse_args()

args = parse_args()

# /export/c11/sli136/l2mt/base_bpe_new/results/l2/generate-test.txt 0.6371
# /export/c11/sli136/l2mt/base_bpe_new/results/l2-valid/generate-test.txt 0.5984
# /export/c11/sli136/l2mt/base_bpe_new/results/ref/generate-test.txt 0.6136
# /export/c11/sli136/l2mt/base_bpe_new/results/ref-valid/generate-test.txt 38.5968

if args.pred_file is not None and args.tgt_file is not None:
    filename = args.pred_file
    with open(args.pred_file, "r") as f:
        hypothesis = [s.strip() for s in f.readlines()]
    with open(args.tgt_file, "r") as f:
        reference = [s.strip() for s in f.readlines()]

else:
    filename = os.path.join(args.dir_name, "generate-test.txt")
    
    # bitext_filename = filename.split('.')[0] + '.bitext'
    pred_file = filename.split('.')[0] + '.pred'
    tgt_file = filename.split('.')[0] + '.tgt'

    # if not os.path.exists(bitext_filename):
    if not os.path.exists(pred_file) or not os.path.exists(tgt_file):
        hypothesis = []
        reference = []
        with open(filename, "r") as f:
            for line in f.readlines():
                if 'T-' in line:
                    reference.append(line.split('\t')[1].strip())
                elif 'D-' in line:
                    hypothesis.append(line.split('\t')[2].strip())

        # with open(bitext_filename, "w+") as f:
        #     for i in range(len(hypothesis)):
        #         f.write(hypothesis[i] + '\t' + reference[i] + '\n')
        with open(pred_file, "w+") as f:
            f.writelines('\n'.join(hypothesis) + '\n')
        with open(tgt_file, "w+") as f:
            f.writelines('\n'.join(reference) + '\n')
    else:
        with open(pred_file, "r") as f:
            hypothesis = [s.strip() for s in f.readlines()]
        with open(tgt_file, "r") as f:
            reference = [s.strip() for s in f.readlines()]

        # hypothesis = []
        # reference = []
        # with open(bitext_filename, "r") as f:
        #     for line in f.readlines():
        #         h, r = line.split('\t')
        #         hypothesis.append(h.strip())
        #         reference.append(r.strip())

reference = [[r] for r in reference]

if args.ter:
    from sacrebleu import corpus_ter
    sacrebleu_ter = corpus_ter(hypothesis, reference, normalized=False, no_punct=False, asian_support=False, case_sensitive=True)
    # print("{} TER: {}".format(filename, sacrebleu_ter))
    print("TER: {}".format(sacrebleu_ter))

if args.sacrebleu:
    import sacrebleu
    sacrebleu_bleu = sacrebleu.corpus_bleu(hypothesis, reference)
    # print("{} sacrebleu: {}".format(filename, sacrebleu_bleu))
    print("sacrebleu: {}".format(sacrebleu_bleu))

if args.nltk:
    from nltk.translate import bleu_score
    sacrebleu_bleu = bleu_score.corpus_bleu(list_of_references=reference, hypotheses=hypothesis)
    # print("{} nltk bleu: {}".format(filename, sacrebleu_bleu))
    print("nltk bleu: {}".format(sacrebleu_bleu))

if args.bleu:
    from fairseq.scoring import bleu
    import sentencepiece as spm
    import torch

    scorer = bleu.Scorer(bleu.BleuConfig())
    sp = spm.SentencePieceProcessor(model_file=str(args.spm))

    reference_spm = sp.Encode([r[0] for r in reference], out_type=int)
    hypothesis_spm = sp.Encode(hypothesis, out_type=int)
    for r, h in zip(reference_spm, hypothesis_spm):
        scorer.add(torch.IntTensor(r), torch.IntTensor(h))
    # print("{} fairseq bleu: {}".format(filename, scorer.result_string()))
    print("spm bleu: {}".format(scorer.result_string()))

if args.bertscore:
    from fairseq.scoring import bertscore
    scorer = bertscore.BertScoreScorer(bertscore.BertScoreScorerConfig(bert_score_lang="es"))
    for r, h in zip(reference, hypothesis):
        scorer.add_string(r[0],h)
    print("bertscore: {}".format(scorer.result_string()))

if args.meteor:
    from fairseq.scoring import meteor
    scorer = meteor.MeteorScorer(args=None)
    for r, h in zip(reference, hypothesis):
        scorer.add_string(r[0].split(),h.split())
    print("meteor: {}".format(scorer.result_string()))