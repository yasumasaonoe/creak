import json

for split in ['train', 'dev', 'test']:
    with open(f'fm2/{split}.jsonl', 'r') as f:
        data = [json.loads(l) for l in f]
    with open(f'fm2/{split}_converted.json', 'w') as f:
        for ex in data:
            ex_id = ex['id']
            converted_ex = {'ex_id': f'fm2_{split}_{ex_id}',
                            'sentence': ex['text'],
                            'explanation': 'n/a',
                            'label': 'true' if ex['label'] == "SUPPORTS" else 'false',
                            'entity': 'n/a',
                            'en_wiki_pageid': 'n/a',
                            'entity_mention_loc': [[0, 0]]}
            f.write(json.dumps(converted_ex))
            f.write('\n')
