mkdir fever_kilt
wget http://dl.fbaipublicfiles.com/KILT/fever-train-kilt.jsonl -P fever_kilt
wget http://dl.fbaipublicfiles.com/KILT/fever-dev-kilt.jsonl -P fever_kilt
python convert_fever.py