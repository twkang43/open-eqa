# Copyright (c) Meta Platforms, Inc. and affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import json
from pathlib import Path

import numpy as np

CATEGORIES = ["object recognition", "object localization", "attribute recognition", "spatial understanding",
              "object state recognition", "functional reasoning", "world knowledge"]

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "metrics",
        type=Path,
        help="path to a metrics file",
    )
    parser.add_argument(
        "--dataset",
        type=Path,
        default="data/open-eqa-v0.json",
        help="path to dataset (default: data/open-eqa-v0.json)",
    )
    parser.add_argument(
        "-f",
        "--force",
        action="store_true",
        help="evaluate results even if responses are missing (default: false)",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="print verbose outputs (default: false)",
    )
    args = parser.parse_args()
    assert args.metrics.exists()
    assert args.dataset.exists()
    return args


def main(args: argparse.Namespace):
    # load metrics
    metrics = json.load(args.metrics.open("r"))
    question_id_to_metrics = metrics
    print("found {:,} metrics".format(len(metrics)))

    # load dataset
    dataset = json.load(args.dataset.open("r"))
    dataset_question_ids = [item["question_id"] for item in dataset]
    question_id_to_item = {item["question_id"]: item for item in dataset}
    print("found {:,} questions".format(len(dataset)))

    # check that results and dataset match
    if not args.force:
        assert len(metrics) == len(dataset_question_ids)
        assert set(metrics) == set(dataset_question_ids)

    # load scores
    all_scores = {}
    if args.metrics.exists():
        all_scores = json.load(args.metrics.open("r"))
        print("found {:,} existing scores".format(len(all_scores)))
        
    # calculate scores for each category
    category_scores = {category: [] for category in CATEGORIES}
    
    for question_id, score in question_id_to_metrics.items():
        if question_id in question_id_to_item:
            category = question_id_to_item[question_id]["category"]
            category_scores[category].append(score)
            
    for category, scores in category_scores.items():
        scores = np.array(scores)
        scores = 100.0 * (np.clip(scores, 1, 5) - 1) / 4
        print("{}: {:.1f}".format(category, np.mean(scores)))

    # calculate final score
    scores = np.array(list(all_scores.values()))
    scores = 100.0 * (np.clip(scores, 1, 5) - 1) / 4
    print("final score: {:.1f}".format(np.mean(scores)))


if __name__ == "__main__":
    main(parse_args())
