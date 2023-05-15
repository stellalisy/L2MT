#! /usr/bin/env python3

import os
import argparse
import random
import numpy as np
# from pathlib import Path

errors_dict = {
    'typo': '/home/sli136/l2mt/data/nmt-grammar-noise/en-artl2-errors',
    'both': '/home/sli136/l2mt/data/nmt-grammar-noise/en-artl2-errors',
    'runon': '/home/sli136/l2mt/data/nmt-grammar-noise/en-artl2-errors',
    'para': '/home/sli136/l2mt/data/nmt-grammar-noise/en-artl2-errors',
    'article': '/home/sli136/l2mt/data/nmt-grammar-noise/en-grammar-errors',
    'dropone': '/home/sli136/l2mt/data/nmt-grammar-noise/en-grammar-errors',
    'nounnum': '/home/sli136/l2mt/data/nmt-grammar-noise/en-grammar-errors',
    'prep': '/home/sli136/l2mt/data/nmt-grammar-noise/en-grammar-errors',
    'sva': '/home/sli136/l2mt/data/nmt-grammar-noise/en-grammar-errors',
}
errors_weights = {
    'typo': 1,
    'both': 1,
    'runon': 1,
    'para': 1,
    'article': 1,
    'dropone': 1,
    'nounnum': 1,
    'prep': 1,
    'sva': 1,
}
CLEAN_DIR = '/home/sli136/l2mt/data/nmt-grammar-noise/en-es-experiments'
OUT_DIR = '/home/sli136/l2mt/data/raw/all-mixed'
# SET_NAME = ['clean', 'dropone']
ERRORS = list(errors_dict.keys())
OUT_NAME = 'all-mixed'
RATIO=3

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "-s",
        "--source_lang",
        type=str,
        default='en',
        help="source directory location"
    )
    parser.add_argument(
        "-t",
        "--target_lang",
        type=str,
        default='es',
        help="source directory location"
    )
    parser.add_argument(
        "-o",
        "--output_dir",
        type=str,
        default=OUT_DIR,
        help="output file directory"
    )
    parser.add_argument(
        "-k",
        "--key",
        type=str,
        default=OUT_NAME,
        help="output file directory"
    )
    parser.add_argument(
        "-r",
        "--ratio",
        type=int,
        default=RATIO,
        help="(approximate) ratio of crappy to clean data"
    )
    parser.add_argument(
        "-e",
        "--errors",
        nargs="*",
        type=str,
        default=ERRORS,
        help="human errors to generate"
    )
    args = parser.parse_args()
    if not os.path.isdir(args.output_dir):
        os.mkdir(args.output_dir)
    return args

# Path(os.path.join(args.output_directory, args.exp_name)).mkdir(parents=True, exist_ok=True)
def open_files(errors_dict, split, lang):
    files = dict()
    for error, path in errors_dict.items():
        filename = split + '.' + error + '.' + lang
        if os.path.exists(os.path.join(path, filename)):
            files[error] = open(os.path.join(path, filename), 'r')
    return files

def close_files(files):
    for f in files.values():
        f.close()

def read_error_lines(files, tgt_files, tgt_line, ratio, src_line=''):
    lines, weights = dict(), []
    for error, f in files.items():
        e_line = f.readline().replace(' .', '.').replace(' ,', ',').replace(' ?', '?').replace(' !', '!').strip()
        if e_line == '':
            continue
        lines[error] = e_line
        weights.append(errors_weights[error])
    
    tgt_dict = dict()
    for error, f in tgt_files.items():
        tgt_dict[error] = f.readline().strip()

    probs = [w/sum(weights) for w in weights]
    ratio = min(int(ratio + random.uniform(-1, 1)), len(lines))
    if ratio < 1: return [], []
    error_idxs = list(np.random.choice(list(lines.keys()), size=ratio, replace=False, p=probs))  # name of the errors we want to add
    error_lines = [lines[i] for i in error_idxs]

    tgt_lines = []
    for err in error_idxs:
        if err in tgt_dict:
            tgt_lines.append(tgt_dict[err])
        else:
            tgt_lines.append(tgt_line.strip())
    
    ret_error_lines, ret_tgt_lines, max_repeat_reached = [], [], False
    src_line = src_line.replace(' .', '.').replace(' ,', ',').replace(' ?', '?').replace(' !', '!').strip()
    for e_line, t_line in zip(error_lines, tgt_lines):
        if e_line == src_line and not max_repeat_reached:
            ret_error_lines.append(e_line)
            ret_tgt_lines.append(t_line)
            max_repeat_reached = True
        elif e_line != src_line:
            ret_error_lines.append(e_line)
            ret_tgt_lines.append(t_line)
    return ret_error_lines, ret_tgt_lines

def main(args):
    # splits = ['train']
    splits = ['train', 'dev','test']
    for split in splits:
        print("Combining data for split: ", split)
        clean_file = os.path.join(CLEAN_DIR, split+'.clean.'+args.source_lang)
        target_file = os.path.join(CLEAN_DIR, split+'.clean.'+args.target_lang)
        src_output = os.path.join(args.output_dir, split+'.'+OUT_NAME+'.'+args.source_lang)
        tgt_output = os.path.join(args.output_dir, split+'.'+OUT_NAME+'.'+args.target_lang)
        with open(src_output, 'w') as f: f.write('')
        with open(tgt_output, 'w') as f: f.write('')

        f_clean = open(clean_file, 'r')
        f_target = open(target_file, 'r')
        f_output_src = open(src_output, 'a+')
        f_output_tgt = open(tgt_output, 'a+')
        err_files = open_files(errors_dict, split, args.source_lang)
        tgt_files = open_files(errors_dict, split, args.target_lang)
        for i, (clean_line, tgt_line) in enumerate(zip(f_clean.readlines(), f_target.readlines())):
            error_lines, tgt_lines = read_error_lines(err_files, tgt_files, tgt_line, args.ratio, src_line=clean_line)
            error_lines += [clean_line.strip()]
            tgt_lines += [tgt_line.strip()]
            append_output_fast(error_lines, tgt_lines, f_output_src, f_output_tgt)
            if i % 1000 == 0:
                print("processed {} lines".format(i))
        f_clean.close()
        f_target.close()
        f_output_src.close()
        f_output_tgt.close()
        close_files(err_files)
        close_files(tgt_files)
            
            # for line in lines:
            #     temp = line.split('\t')
            #     if len(temp) != 2 or temp[1].strip() == "":
            #         continue
            #     sent_id = temp[0]
            #     dataDict[sent_id] = {src_lang: temp[1].strip(), tgt_lang: ""}
                
    # print("\ninference dataset size:", len(dataDict))
    # print("sample sent:", dataDict[list(dataDict.keys())[50]])
    # return dataDict

def write_mixed(mixed_input, target_output, split, args):
    # final_src = []
    # final_tgt = []
    filename_src = split + '.' + OUT_NAME + '.' + args.source_lang
    filename_tgt = split + '.' + OUT_NAME + '.' + args.target_lang
    # num_repeat = len(mixed_input)
    f_src = open(os.path.join(args.output_dir, filename_src), 'w+')
    f_tgt = open(os.path.join(args.output_dir, filename_tgt), 'w+')
    for repeat in range(len(target_output)):
        temp = list(set([v[repeat] for v in mixed_input.values()]))
        rand_out = random.sample(temp, args.num_repeat)
        f_src.writelines('\n'.join(rand_out) + '\n')
        f_tgt.writelines('\n'.join([target_output[repeat] for _ in range(len(rand_out))]) + '\n')
        # final_src.extend(temp)
        # final_tgt.extend([target_output[repeat] for _ in range(num_repeat)])

def append_output_fast(error_lines, tgt_lines, f_output_src, f_output_tgt):
    f_output_src.writelines('\n'.join(error_lines) + '\n')
    f_output_tgt.writelines('\n'.join(tgt_lines) + '\n')

def random_ref(args, mode):
    all_ref = []
    for i in range(4):
        with open(os.path.join(args.data_dir, mode, mode+'.ref'+str(i)), 'r') as f:
            all_ref.append([s.strip() for s in f.readlines()])
    rand_out = []
    
    for i in range(len(all_ref[0])):
        rand_out.append(all_ref[random.randint(0,3)][i])
    with open(os.path.join(args.output_dir, mode+'.en-es.en'), 'w') as f:
        f.writelines('\n'.join(rand_out))

if __name__ == "__main__":
    args = parse_args()
    main(args)
    # random_ref(args, 'dev')
    # random_ref(args, 'test')