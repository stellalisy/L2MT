import argparse
import os
import random
import myconstants
import numpy as np
import string
import re
import logging
import nltk
from nltk.tag import pos_tag
import math 

SOURCE_LANG = 'en'
TARGET_LANG = 'es'

##############################################################################
# Helper functions: command line parser
##############################################################################
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "-s",
        "--source-file",
        type=str,
        default='/home/sli136/l2mt/data/nmt-grammar-noise/en-es-experiments/train.clean',
        help="source file prefix"
    )
    parser.add_argument(
        "-t",
        "--target-output",
        type=str,
        default='/home/sli136/l2mt/data/nmt-grammar-noise/en-l2-errors/train.artl2',
        help="target output prefix"
    )
    parser.add_argument(
        "--tgt-lang",
        type=str,
        default=TARGET_LANG,
        help="target language"
    )
    parser.add_argument(
        "-b",
        "--batch_size",
        type=int,
        default=8,
        help="batch size"
    )
    parser.add_argument(
        "-l",
        "--logging",
        type=str,
        default='/home/sli136/l2mt/output/cpu-shell/mylog/',
        help="target output prefix (can be a directory or a file name)"
    )
    parser.add_argument(
        "-ce",
        "--compound-errors",
        action=argparse.BooleanOptionalAction,
        type=bool,
        default=False,
        help="compound the errors (when more than 1 error types are selected)"
    )
    parser.add_argument(
        "--dry-run",
        action=argparse.BooleanOptionalAction,
        type=bool,
        default=False,
        help="dry run, no output"
    )
    parser.add_argument(
        "-e",
        "--errors",
        nargs="*",
        type=str,
        default=["runon_comb", "typo", "runon_comb"],
        help="human errors to generate"
    )
    parser.add_argument(
        "-p",
        "--probabilities",
        nargs="*",
        type=float,
        default=[0.4, 0.2, 0.1],
        help="human errors to generate"
    )
    return parser.parse_args()
args = parse_args()

##############################################################################
# Helper functions: set up
##############################################################################
def setup_logger(log_file, run_name):
    if os.path.isdir(log_file):  # if user provided log path is a directory
        log_file = os.path.join(log_file, run_name + '.'+SOURCE_LANG + '.log')
    elif not os.path.isabs(log_file):  # if user provided log path is a not an absolute path
        os.makedirs(os.path.join(os.getcwd(), os.path.dirname(log_file)), exist_ok=True)
        log_file = os.path.join(os.getcwd(), log_file)
    logging.basicConfig(filename=log_file, filemode='w+', encoding='utf-8', 
                        format='%(asctime)s %(message)s', datefmt='[%Y-%m-%d %H:%M:%S]',
                        level=logging.INFO)
    return log_file

run_name = args.target_output.split('/')[-1]
args.logging = setup_logger(args.logging, run_name)
src_output = args.target_output + '.' + SOURCE_LANG
trans_output = args.target_output + '.' + args.tgt_lang
train_test_dev_split = run_name.split('.')[0]



for arg in vars(args):
    logging.info(f'{arg}: {getattr(args, arg)}')
    print(f'{arg}: {getattr(args, arg)}')
for error in args.errors:
    if error not in myconstants.ERRORS:
        raise ValueError("Error type %s not supported" % error)
if len(args.errors) != len(args.probabilities):
    raise ValueError("Number of errors and probabilities do not match")
if "runon_comb" in args.errors and args.probabilities[args.errors.index("runon_comb")] == 1:
    raise ValueError("Probability of runon_comb cannot be 1.0")

if "paraphrase" in args.errors:
    from parrot import Parrot
    import warnings
    warnings.filterwarnings("ignore")
    parrot = Parrot(model_tag="prithivida/parrot_paraphraser_on_T5")

##############################################################################
# Helper functions: paraphrase errors
##############################################################################
def paraphrase_line(tokens: list[str]) -> tuple[list[str], int]:
    """
    paraphrase_line: generate line-level paraphrase errors
        for all occurances of a word in the paraphrase dictionary, 
        randomly decide whether to replace it with a paraphrase
    """
    if 'paraphrase' not in args.errors:
        return tokens, 0
    # print("-"*100)
    phrase = ' '.join(tokens)
    print()
    print("Input_phrase:", phrase)
    # p = args.probabilities[args.errors.index("paraphrase")]
    para_phrases = parrot.augment(input_phrase=phrase, use_gpu=False, max_return_phrases=5, do_diverse=True, adequacy_threshold=0.5, fluency_threshold=0.2)  # 0.7, 0.4 for half and 0.5, 0.2 for half
    if not para_phrases:
        print("No paraphrase found")
        # print("-"*100)
        return tokens, 0
    para_phrases = [p[0] for p in para_phrases]
    weights = [1 if i>0 else 0.3 for i in range(len(para_phrases))]
    # print("para phrases:", para_phrases)
    para_phrase = random.choices(para_phrases, weights=weights, k=1)[0]  # weights=[p**(i/2) for i in range(len(para_phrases))]
    try:
        if len(para_phrase) == 0:
            print("No paraphrase found")
            # print("-"*100)
            return tokens, 0
        para_phrase = para_phrase.capitalize()
        if phrase[-1] in string.punctuation and para_phrase[-1] not in string.punctuation:
            if phrase[0] == '"' and phrase[-1] == '"':
                if phrase[-2] in string.punctuation and para_phrase[-1] not in string.punctuation:
                    para_phrase = '"' + para_phrase + phrase[-2] + '"'
                else:
                    para_phrase = '"' + para_phrase + '"'
            if phrase[0] == '(' and phrase[-1] == ')':
                if phrase[-2] in string.punctuation and para_phrase[-1] not in string.punctuation:
                    para_phrase = '(' + para_phrase + phrase[-2] + ')'
                else:
                    para_phrase = '(' + para_phrase + ')'
            else:
                para_phrase += phrase[-1]
        if phrase == para_phrase:
            print("Paraphrased sentence is the same as the original sentence")
            # print("-"*100)
            return tokens, 0
        print("Para_phrase :", para_phrase)
        # print("-"*100)
        return para_phrase.split(), 1
    except:
        print("No paraphrase found")
        # print("-"*100)
        return tokens, 0


##############################################################################
# Helper functions: runOn errors
##############################################################################
def find_all(arr: list[str], val: str) -> list[int]:
    """
    find_all: find all indices of a value in a list
    input: arr, val
        arr: tokens (list of strings)
        val: element of interest (string)
    output: list of indices
    """
    return [1 if value == val and idx<len(arr)-1 else 0 for idx, value in enumerate(arr)]

def runOn_line(tokens: list[str]) -> tuple[list[str], int]:
    """
    runOn_line: generate line-level run-on errors
        for all occurances of '.' not at the end of the line, 
        randomly decide whether convert it to a run-on error (by replacing it with a conjunction)
        if yes, decide whether to lowercase the next token
    input: tokens (list of strings)
    output: tokens (list of strings), num_changes (int)
    """
    if 'runon_line' not in args.errors:
        return tokens, 0
    potentials = find_all(tokens, '.') # find all indices of '.'
    i, num_changes, ret = 0, 0, ''
    while i < len(tokens):
        if potentials[i] == 1:
            if random.random() < args.probabilities[args.errors.index("runon_line")]:
                conj = random.choice(myconstants.CONJUNCTIONS)
                next = runOn_append(tokens[i+1:])[0]
                ret = ret.strip() + conj + ' ' + next + ' '
                i += 2
                num_changes += 1
            else:  # if not selected, just add '.' to the end of sentence (remove the space before it)
                ret = ret.strip() + tokens[i] + ' '
                i += 1
        else:  # add the token to the end of sentence
            ret += tokens[i] + ' '
            i += 1
    tokens = ret.strip().split()
    return tokens, num_changes

def runOn_append(tokens: list[str]) -> list[str]:
    """
    runOn_append: process the next sentence (list of tokens) after a run-on creation
        if the first token is a proper noun, capitalize it
        if the first token is "I", leave it as it is
        otherwise, lowercase it
    input: tokens (list of strings)
    output: tokens (list of strings)
    """
    if tokens[0] == 'I':
        return tokens[0]
    tagged_sent = pos_tag(tokens)
    propernouns = ['NNP', 'NNPS']
    if tagged_sent[0][1] in propernouns:
        return [tokens[0].capitalize()] + tokens[1:]
    return [tokens[0].lower()] + tokens[1:]

def runOn_prepend(sent: str) -> str:
    """
    runOn_prepend: process the previous sentence (string) before a run-on creation
        'I run' -> 'I run,' (or ';', or ', and')
        'I run.' -> 'I run,' (or ';', or ', and')
        'I run!' -> 'I run,' (or ';', or ', and')
        'I run?' -> 'I run,' (or ';', or ', and')
        'I "run"' -> 'I "run,"' (or ';', or ', and')
    """
    prev_tokens = str_to_tokens(sent)
    # print("line 186 in runOn_prepend(), prev_tokens:", prev_tokens)
    last_token = list(prev_tokens[-1])
    # change the punctuation of the last token of the previous sentence to a conjunction
    if last_token[-1] not in string.punctuation:  # if the last token is not a punctuation
        prev_tokens[-1] = ''.join(last_token) + random.choice(myconstants.CONJUNCTIONS)
    elif (last_token[-1] == '"' or last_token[-1] == "'") and len(last_token) > 1: 
        if last_token[-2] in string.punctuation:  # if the token has the form [a-zA-Z]+[.,?!;:]["']
            prev_tokens[-1] = ''.join(last_token[:-2]) + random.choice([",", ",", ";"]) + last_token[-1]
        else:  # if the token has the form [a-zA-Z]+["']
            conj = random.choice(myconstants.CONJUNCTIONS)
            if conj == " and":
                prev_tokens[-1] = ''.join(last_token) + conj
            elif conj == ", and":
                prev_tokens[-1] = ''.join(last_token[:-1]) + ',' + last_token[-1] + ' and'
            else:
                prev_tokens[-1] = ''.join(last_token[:-1]) + conj + last_token[-1]
    else:  # if the token has the form [a-zA-Z]+[.,?!;:]
        prev_tokens[-1] = ''.join(last_token[:-1]) + random.choice(myconstants.CONJUNCTIONS)
    return " ".join(prev_tokens)
    
def runOn_comb(prev_en: list[str], prev_es: list[str], tokens: list[str], line_es: str, prob: float) -> tuple[list[str], list[str], int]:
    """
    runOn_comb: combine the previous sentence with the current sentence
    input: prev_en, prev_es, tokens, line_es
        prev_en: list of previous english sentences (strings)
        prev_es: list of previous spanish sentences (strings)
        tokens: current english sentence (list of words)
        line_es: current spanish sentence (string)
    output: prev_en, prev_es, num_written
        prev_en: updated list of previous english sentences (strings)
        prev_es: updated list of previous spanish sentences (strings)
        num_written: number of (original) lines written to 1 line of the output file
    """
    # no runon_comb, write the cached previous sentence pair(s), update the cache, return cache
    if random.random() > prob:
        num_out = write_output(prev_en, prev_es)
        prev_en = [" ".join(tokens)]
        prev_es = [line_es.strip()]
        # print("(prob={}): write the cached {} sentences into one line".format(prob, num_out))
        return prev_en, prev_es, num_out
    else:
        assert len(prev_en) == len(prev_es) and len(prev_en) > 0  # sanity check to make sure the cache is not empty
        prev_en[-1] = runOn_prepend(prev_en[-1])
        prev_en.append(" ".join(runOn_append(tokens)))
        prev_es.append(line_es.strip())
        # print("(prob={}): append to cache, now {} sentences in cache".format(prob, len(prev_es)))
        return prev_en, prev_es, 0

def runOn_concat(prev_en: list[str], prev_es: list[str], tokens: list[str], line_es: str) -> tuple[list[str], list[str]]:
    assert len(prev_en) == len(prev_es) and len(prev_en) > 0  # sanity check to make sure the cache is not empty
    prev_en[-1] = runOn_prepend(prev_en[-1])
    prev_en.append(" ".join(runOn_append(tokens)))
    prev_es.append(line_es.strip())
    # print("(prob={}): append to cache, now {} sentences in cache".format(prob, len(prev_es)))
    return prev_en, prev_es

def write_output(prev_en: list[str], prev_es: list[str]) -> int:
    """
    write_output: write the cached previous sentence pair(s) to the output file
    input: prev_en, prev_es
        prev_en: list of previous english sentences (strings)
        prev_es: list of previous spanish sentences (strings)
    output: num_written
        num_written: number of (original) lines written to 1 line of the output file
    """
    assert len(prev_en) == len(prev_es) and len(prev_en) > 0  # sanity check to make sure the cache is not empty
    if not args.dry_run:
        with open(src_output, "a+") as g:
            g.write(" ".join(prev_en) + "\n")
        with open(trans_output, "a+") as g:
            g.write(" ".join(prev_es) + "\n")
    return len(prev_en)
    
############################################################################## 
# Helper functions: typo errors
##############################################################################
def substitute_case(letter):
    temp_letter = random.choices(myconstants.ALPHABET, weights=myconstants.TRANSITION_PROB[myconstants.ALPH_TO_IDX[letter.lower()]], k=1)[0]
    if letter.lower() != letter:
        return temp_letter.upper()
    else:
        return temp_letter

def typo(token): 
    """create typo in token, deprecated"""
    # old_tokens = list(token)
    if 'typo' not in args.errors:
        return token, 0, 0
    token_list = list(token)
    min_len = 6
    if len(token_list) < min_len:
        return token, 0, 0
    # estimated probability of a typo in a token is prob / min_len * 2
    if random.random() > (args.probabilities[args.errors.index("typo")] * len(token_list) / 10):
        return token, 0, 0
    return typo_word(token)

def typo_word(token):
    """
    Create typo in one token
    token is already alphabetic, no need to check
    already decided to mutate token, no need to check
    input: token can be a string or a list of characters
    output: token, num_typos, num_swaps"""
    token_list = list(token)
    if random.random() < 0.5:  # use typo table (substitution error typo)
        weight = [0 if letter.lower() not in myconstants.ALPHABET else myconstants.LETTER_WEIGHT[myconstants.ALPH_TO_IDX[letter.lower()]] for letter in token_list]
        if sum(weight) == 0: # all characters are non alphabetic, skip
            return token, 0, 0
        idx = random.choices(range(len(token_list)), weights=weight, k=1)[0] # randomly select a letter
        token_list[idx] = substitute_case(token_list[idx])
        return ''.join(token_list), 1, 0
    else:  # swapping error typo
        weight = [1 if ch.isalpha() else 0 for ch in token_list] # do not need to check for whether all characters are non alphabetic
        weight[np.max(np.nonzero(weight))] = 0
        if sum(weight) == 0:  # can leave it here but basically useless, checked in parent function
            return token, 0, 0
        idx = random.choice(range(len(token_list)-1))
        token_list[idx], token_list[idx+1] = token_list[idx+1], token_list[idx]
        return ''.join(token_list), 0, 1

def typo_sent(tokens):
    if 'typo' not in args.errors:
        return tokens, 0, 0
    # add typos to a sentence (tokens)
    # number of typos equals the minimum between 1 and the number of alphabetic tokens * typo_prob
    min_sent_len = 4
    min_word_len = 6
    # pre-check for sentence length and alphabetic tokens
    if len(tokens) < min_sent_len:  # sentence is too short, skip
        return tokens, 0, 0
    if sum([1 if t.translate(str.maketrans('', '', string.punctuation)).isalpha() else 0 for t in tokens]) == 0:  # sentence does not have alphabetic tokens, skip
        return tokens, 0, 0
    
    # set number of typos based on indicated probability and number of long-enough alphabetic tokens
    num_typos = int(len(tokens) * args.probabilities[args.errors.index("typo")])
    if num_typos < 1: num_typos = 1
    # check whether word length is long enough (only counting alphabetic characters), do not allow alphanumeric words (e.g. 1st)
    word_length_check = [1 if (len(t.translate(str.maketrans('', '', string.punctuation)))>=min_word_len and t.translate(str.maketrans('', '', string.punctuation)).isalpha()) else 0 for t in tokens]
    if sum(word_length_check) == 0:  # no token is long enough, skip entire sentence
        return tokens, 0, 0
    if sum(word_length_check) < num_typos:  # number of long-enough words in the sentence not enough, reduce number of typos
        num_typos = sum(word_length_check)
    
    # randomly select tokens to add typos, longer words have higher probability of being selected (do not count punctuations)
    idxs = random.choices(range(len(tokens)), weights=[len(t.translate(str.maketrans('', '', string.punctuation))) * word_length_check[i] for i, t in enumerate(tokens)], k=num_typos)
    num_typo_sub_local, num_typo_swap_local = 0, 0
    for idx in idxs:
        tokens[idx], typo_sub, typo_swap = typo_word(tokens[idx])
        num_typo_sub_local += typo_sub
        num_typo_swap_local += typo_swap
    return tokens, num_typo_sub_local, num_typo_swap_local


##############################################################################
# Helper functions: lower level helper functions
##############################################################################
def tokens_to_str(tokens: list[str]):
    return " ".join(tokens)

def str_to_tokens(line: str):
    return line.strip().split()

def sentence_end(token):
    if token[-1] != '.':  # no punctuation in token
        return False
    if token.endswith(myconstants.ACRONYMS) or re.search('[A-Z].[A-Z].', token):  # exceptions listed in myconstants or matches the X.Y. patten
        return False
    if re.search('[A-Z].', token): # matches the X. pattern
        return False
    return True

def preprocess(sent):
    """Preprocess a sentence.
    Args: sent (str): a sentence
    Returns: tokens (list): a preprocessed tokens list"""
    # remove extra spaces
    sent = re.sub(' +', ' ', sent)
    # remove leading and trailing spaces
    sent_split = sent.strip().split()
    sent_p = ''
    for i, token in enumerate(sent_split):
        # remove punctuations
        if i < len(sent_split)-1 and sentence_end(token):
            token += ' '
            sent_p += token.replace('. ', ' . ')
        else:
            sent_p += token
        sent_p += ' '
    return sent_p.strip().rstrip().split()

def get_report(all_stats):
    return [
        "==================================================================================",
        f"Generating errors ({','.join(args.errors)}) for the {train_test_dev_split} set:",
        f"Number of lines with in-line changes: {all_stats.num_inline_chg} out of {all_stats.total_line_after} lines ({all_stats.num_inline_chg/all_stats.total_line_after*100}% of total lines)",
        f"Number of lines w multi-line concats: {all_stats.num_multiline_chg} out of {all_stats.total_line_after} final total lines ({all_stats.num_multiline_chg/all_stats.total_line_after*100}% of total lines)\n",
        f"Number of sub  typos created: {all_stats.num_typo_sub} out of {all_stats.total_token_after} tokens ({int(all_stats.num_typo_sub/all_stats.total_line_after)} per sentence, {(all_stats.num_typo_sub)/all_stats.total_token_after*100}% of total tokens)",
        f"Number of swap typos created: {all_stats.num_typo_swap} out of {all_stats.total_token_after} tokens ({int(all_stats.num_typo_swap/all_stats.total_line_after)} per sentence, {(all_stats.num_typo_swap)/all_stats.total_token_after*100}% of total tokens)",
        f"Number of all  typos created: {all_stats.num_typo_sub+all_stats.num_typo_swap} out of {all_stats.total_token_after} tokens ({int((all_stats.num_typo_sub+all_stats.num_typo_swap)/all_stats.total_line_after)} per sentence, {(all_stats.num_typo_sub+all_stats.num_typo_swap)/all_stats.total_token_after*100}% of total tokens)\n",
        f"Number of paraphrased sentences created: {all_stats.num_paraphrase} out of {all_stats.total_line_after} lines ({all_stats.num_paraphrase/all_stats.total_line_after*100}% of total lines)",
        f"Number of line-level run-on errors created: {all_stats.num_runOn_line} out of {all_stats.total_line_after} lines ({all_stats.num_runOn_line/all_stats.total_line_after*100}% of total lines)",
        f"Number of multi-line run-on errors created: {all_stats.num_runOn_comb} - average is {all_stats.num_runOn_comb/all_stats.total_line_after} sentences per line",
        f"Number of non-alphabetic tokens: {all_stats.non_alpha} out of {all_stats.total_token_before} tokens ({all_stats.non_alpha/all_stats.total_token_before*100}% of total tokens)\n",
        f"Avg. sent  length: {int(all_stats.total_token_before/all_stats.total_line_before)} ==> {int(all_stats.total_token_after/all_stats.total_line_after)} tokens",
        f"Total token count: {all_stats.total_token_before} ==> {all_stats.total_token_after} tokens",
        f"Total line  count: {all_stats.total_line_before} ==> {all_stats.total_line_after} lines",
        f"Artificial error written to {src_output}",
        f"Corresponding {args.tgt_lang} translations written to {trans_output}",
        f"Log written to {args.logging}",
        "=================================================================================="]

def display_report(func, all_stats, num_lines=-1):
    for i, line in enumerate(get_report(all_stats)):
        if i == 1:
            if num_lines < 0: line = f'[DONE] ' + line 
            else: line = f'[IN PROGRESS: 0-{num_lines} lines] ' + line
        func(line)


##############################################################################
# helper classes
##############################################################################
class Stats:
    def __init__(self):
        self.num_runOn_line = 0     # number of periods that have been changed to conjunctions
        self.num_runOn_comb = 0     # number of sentences have been added after another sentence
        self.num_typo_sub = 0       # number of typos that have been substitution errors
        self.num_typo_swap = 0      # number of typos that have been swap errors
        self.num_paraphrase = 0
        self.num_inline_chg = 0     # number of sentences that only considers in-line edits
        self.num_multiline_chg = 0  # number of sentences that only considers multi-line edits
        self.total_token_before = 0 # number of tokens in source data
        self.total_line_before = 0  # number of lines in source data
        self.total_token_after = 0  # number of tokens in source data after adding errors
        self.total_line_after = 0   # number of lines in source data after adding errors
        self.non_alpha = 0          # number of non-alphabetic tokens
all_stats = Stats()


##############################################################################
# batch processing
##############################################################################
def process_batch(batch: list[tuple[str]]):
    all_stats.total_line_before += len(batch)
    tokenized_batch = [(preprocess(line_en), line_es.strip()) for line_en, line_es in batch]
    tokenized_batch_good = [(en, es) for en, es in tokenized_batch if len(en) > 0 and len(es) > 0]

    batch_size = len(tokenized_batch_good)
    if "runon_comb" in args.errors:
        p = args.probabilities[args.errors.index("runon_comb")]
        probs = [p**i for i in range(1, batch_size)]
    
    batch_out_en = []
    batch_out_es = []
    for line_i, (tokens, line_es) in enumerate(tokenized_batch_good):
        processed, num_inline = process_line(tokens, before=True)
        
        ret_sents_en = [tokens_to_str(processed)]  # list[str] of length 1
        ret_sents_es = [line_es.strip()]  # list[str] of length 1

        if "runon_comb" in args.errors:
            other_idxs = [i for i in list(range(batch_size)) if i != line_i]
            num_append = sum([int(random.random() < p) for p in probs])
            append_order = random.sample(other_idxs, k=num_append)
            for sent_id in append_order:
                tok_en, str_es = tokenized_batch_good[sent_id]
                sent_claus, _ = process_line(tok_en)[0] if args.compound_errors else tok_en
                ret_sents_en, ret_sents_es = runOn_concat(ret_sents_en, ret_sents_es, sent_claus, str_es)
            ret_sents_en = [' '.join(ret_sents_en)]
            ret_sents_es = [' '.join(ret_sents_es)]
            all_stats.num_multiline_chg += int(num_append > 0)  # number of sentences that only considers multi-line edits
            all_stats.num_runOn_comb += num_append + 1

        if "paraphrase" in args.errors and len(args.errors) == 1:
            assert len(ret_sents_en) == 1 and len(ret_sents_es) == 1
        batch_out_en.extend(ret_sents_en)
        batch_out_es.extend(ret_sents_es)
        # num_written = write_output(ret_sents_en, ret_sents_es)

        all_stats.num_inline_chg += num_inline     # number of sentences that only considers in-line edits
        all_stats.total_line_after += 1
        all_stats.total_token_after += sum([len(s.split()) for s in ret_sents_en])
    num_written = write_output_batch(batch_out_en, batch_out_es)

    print("batch size:", batch_size, "num_written:", num_written)
    logging.info(f"batch size: {batch_size} num_written: {num_written}")

def write_output_batch(batch_out_en: list[str], batch_out_es: list[str]):
    assert len(batch_out_en) == len(batch_out_es) and len(batch_out_en) > 0  # sanity check to make sure the cache is not empty
    if not args.dry_run:
        src_output_f.write("\n".join(batch_out_en) + "\n")
        trans_output_f.write("\n".join(batch_out_es) + "\n")
    return len(batch_out_en)

def process_line(tokens: list[str], before=False):
    all_stats.total_token_before += sum([0 if all(ch in string.punctuation for ch in token) else 1 for token in tokens]) if before else 0
    all_stats.non_alpha += sum([0 if t.translate(str.maketrans('', '', string.punctuation)).isalpha() else 1 for t in tokens]) # do not need this

    # create possible typo errors
    tokens, d1, d2 = typo_sent(tokens)
    all_stats.num_typo_sub += d1
    all_stats.num_typo_swap += d2

    # create possible run-on errors
    tokens, d3 = runOn_line(tokens)
    all_stats.num_runOn_line += d3

    # create possible paraphrase errors
    tokens, d4 = paraphrase_line(tokens)
    all_stats.num_paraphrase += d4

    tokens = ' '.join(tokens).replace(' . ', '. ').split()  # remove spaces before '.' to get to original form
    
    return tokens, d1+d2+d3+d4


##############################################################################
# main function
##############################################################################
# clear output files
# if not args.dry_run:
#     print("Clearing output files...")
#     with open(src_output, "w+") as g:
#         g.write("")
#     print("[{}]... Done".format(src_output))
#     with open(trans_output, "w+") as g:
#         g.write("")
#     print("[{}]... Done".format(trans_output))

with open(src_output, "r") as f:
    lines_written_actual = len(f.readlines())
lines_written = math.ceil(lines_written_actual // 1000) * 1000
print("Lines written: {}/{}".format(lines_written, lines_written_actual))
logging.info(f"Lines written: {lines_written}/{lines_written_actual}")

with open(args.source_file + '.' + SOURCE_LANG, "r") as f:
    source_lines = len(f.readlines())
if lines_written >= source_lines:
    print("No new lines to process. Exiting...")
    logging.info("No new lines to process. Exiting...")
    exit()

f_en = open(args.source_file + '.' + SOURCE_LANG, "r")
f_es = open(args.source_file + '.' + args.tgt_lang, "r")
batch = []
for line_i, (line_en, line_es) in enumerate(zip(f_en.readlines(), f_es.readlines())):
    if line_i < lines_written:
        continue
    batch.append((line_en, line_es))
    if len(batch) == args.batch_size:
        src_output_f = open(src_output, "a+")
        trans_output_f = open(trans_output, "a+")
        process_batch(batch)
        batch = []
        src_output_f.close()
        trans_output_f.close()
    if (line_i+1) % (6250 * args.batch_size) == 0:
        display_report(print, all_stats, num_lines=line_i+1)

# process last batch
process_batch(batch)

f_en.close()
f_es.close()



display_report(logging.info, all_stats)
display_report(print, all_stats)