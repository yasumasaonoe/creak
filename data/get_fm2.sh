mkdir fm2
wget https://raw.githubusercontent.com/google-research/fool-me-twice/main/dataset/train.jsonl -P fm2
wget https://raw.githubusercontent.com/google-research/fool-me-twice/main/dataset/dev.jsonl -P fm2
wget https://raw.githubusercontent.com/google-research/fool-me-twice/main/dataset/test.jsonl -P fm2
python convert_fm2.py