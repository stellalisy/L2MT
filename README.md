# Robust NMT for L2 Disfluency

Silver dataset creation project using Wikiann data for Named Entity Recognition (NER).

### Step-by-step Instruction

```
# Example:
src_lang=ca
tgt_lang=es
trans_dir=[path/to/translation/directory]
```

1. First, generate errors from clean parallel data:  `qsub scripts/gen_errors.sh [train|dev|test] [typo|runon|paraphrase]`
    ```
    python3 scripts/gen_errors.py \
        -s [path/to/clean/file/prefix] \
        -t [path/to/output/file/prefix] \
        -l [path/to/log/file] \
        --compound-errors \
        -e typo runon paraphrase \
        -p 0.4 0.5 0.7 \
        --tgt-lang es
    ```
2. Second (optional), combine different error types to one dataset: `qsub scripts/move_data.sh`
    ```
    python scripts/move_data.py \
        --ratio 4 \
        --output_dir [path/to/output/directory] \
        --key [experiment-name] \
        --errors article nounnum prep sva typo runon paraphrase
    ```
3. Third, combine error data with clean data to required format: `./scripts/combine_data.sh`
   
4. Train transformers NMT model with fairseq: `qsub scripts/clean_bpe.sh [0|1] [0|1] [0|1] [0|1]`
    ```
    # 1. Tokenize data
    # 2. Fairseq preprocess data
    # 3. Fairseq train
    # 4. Fairseq evaluate
    ```

(All data and models are stored on the clsp grid: /export/c11/sli136/l2mt/)