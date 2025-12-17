python3 serverAI/training/train_intent.py \
 --train serverAI/data/nlu/train1.tsv \
 --valid serverAI/data/nlu/valid1.tsv \
 --out serverAI/models/nlu_intent

python3 serverAI/training/train_ner.py \
--data serverAI/data/nlu/ner_train.json \
--output serverAI/models/ner_model \
--iter 30

build_index: python3 -m serverAI.features.build_index

python3 serverAI/training/train_ranker.py \
    --queries serverAI/eval/queries.jsonl \
    --judgments serverAI/eval/judgments.jsonl \
    --cache serverAI/.cache \
    --out serverAI/models/ranker

  python3 serverAI/tools/labeling_cli.py \
    --seed serverAI/eval/queries_seed.txt \
    --outq serverAI/eval/queries.jsonl \
    --outj serverAI/eval/judgments.jsonl \
    --topk 3

requirements:
    sentencepiece
    torch 
    transformers 
    datasets 
    accelerate 
    seqeval
