import json

for split in ['train', 'dev']:
    with open(f'fever_kilt/fever-{split}-kilt.jsonl', 'r') as f:
        data = [json.loads(l) for l in f]
    with open(f'fever_kilt/{split}_converted.json', 'w') as f:
        for ex in data:
            ex_id = ex['id']
            ans = list(set([p['answer'] for p in ex['output']]))
            assert len(ans) == 1, ans
            ans = ans[0]
            converted_ex = {'ex_id': f'fever_{split}_{ex_id}',
                            'sentence': ex['input'],
                            'explanation': 'n/a',
                            'label': 'true' if ans == "SUPPORTS" else 'false',
                            'entity': 'n/a',
                            'en_wiki_pageid': 'n/a',
                            'entity_mention_loc': [[0, 0]]}
            f.write(json.dumps(converted_ex))
            f.write('\n')
