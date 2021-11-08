# CREAK: A Dataset for Commonsense Reasoning over Entity Knowledge

This repository contains the data and code for the baseline described in the following paper:

> [**CREAK: A Dataset for Commonsense Reasoning over Entity Knowledge**](https://openreview.net/pdf?id=mbW_GT3ZN-)<br/>
> Yasumasa Onoe, Michael J.Q. Zhang, Eunsol Choi, Greg Durrett<br/>
> NeurIPS 2021 Datasets and Benchmarks Track
```
@article{onoe2021creak,
  title={CREAK: A Dataset for Commonsense Reasoning over Entity Knowledge},
  author={Onoe, Yasumasa and Zhang, Michael J.Q. and Choi, Eunsol and Durrett, Greg},
  journal={OpenReview},
  year={2021}
}
```

**\*\*\*\*\* [New] November 8th, 2021: The contrast set has been updated. \*\*\*\*\***

We have increased the size of the contrast set to 500 examples. Please check [the paper](https://openreview.net/pdf?id=mbW_GT3ZN-) for new numbers.


## Datasets

### Examples

![Exampls](./img/fig1.png)

### Data Files
CREAK data files are located under `data/creak`.

- `train.json` contains 10,176 training examples.
- `dev.json` contains 1,371 development examples.
- `test_without_labels.json` contains 1,371 test examples (labels are not included).
- `contrast_set.json` contains 500 contrastive examples.

The data files are formatted as jsonlines. Here is a single training example:
```
{
    'ex_id': 'train_1423',
    'sentence': 'Lauryn Hill separates two valleys as it is located between them.',
    'explanation': 'Lauren Hill is actually a person and not a mountain.',
    'label': 'false',
    'entity': 'Lauryn Hill',
    'en_wiki_pageid': '162864',
    'entity_mention_loc': [[0, 11]]
}
```

| Field                     | Description                                                                              |
|---------------------------|------------------------------------------------------------------------------------------|
| `ex_id`                   | Example ID                                                                               |
| `sentence`                | Claim                                                                                    |
| `explanation`             | Explanation by the annotator why the claim is TRUE/FALSE                                 |
| `label`                   | Label: 'true' or 'false'                                                                 |
| `entity`                  | Seed entity                                                                              |
| `en_wiki_pageid`          | English Wikipedia Page ID for the seed entity                                            |
| `entity_mention_loc`      | Location(s) of the seed entity in the claim                                              |


## Baselines

See [this README](baselines/README.md) 

## Leaderboards

https://www.cs.utexas.edu/~yasumasa/creak/leaderboard.html

We host results only for Closed-Book methods that have been finetuned on only In-Domain data.

To submit your results, please send your system name and prediction files for the dev, test, and contrast sets to `yasumasa@utexas.edu`.


## Contact 

Please contact at `yasumasa@utexas.edu` if you have any questions.