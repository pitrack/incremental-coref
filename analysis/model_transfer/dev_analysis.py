"""
This performs the simulated early stopping experiment (Figure 3 in the
EMNLP 2021 paper).
"""

import sys
import os
import random
import json
import numpy as np
import logging
from collections import defaultdict, Counter
from multiprocessing import Pool
sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir)))
from util import mention_maps
from metrics import CorefEvaluator

logging.basicConfig(
  level=logging.INFO,
  format="%(asctime)s [%(levelname)s] %(message)s",
  datefmt = "%Y-%m-%d %H:%M:%S",
  handlers = [
    logging.FileHandler(sys.argv[3]),
    logging.StreamHandler()
  ]
)


SEED = 0
NUM_SAMPLES = 20
random.seed(SEED)

def process_pred(args):
  (dev_keys, test_keys, epoch) = args
  dev_predictions = {}
  test_predictions = {}
  pred_jsons = [json.loads(doc) for doc in open(f"{checkpoint_path}/preds_{epoch}.json", 'r')]
  for pred_json in pred_jsons:
    doc_key = pred_json["doc_key"]
    if doc_key in dev_keys:
      dev_predictions[doc_key] = pred_json
    elif doc_key in test_keys:
      test_predictions[doc_key] = pred_json
  return (dev_predictions, test_predictions)

def process_preds(checkpoint_path, preds, dev_keys, test_keys):
  max_pred = max([int(x.split("_")[1].split(".")[0]) for x in preds])
  worker_args = [(dev_keys, test_keys, i) for i in range(max_pred + 1)]
  with Pool(processes=20) as pool:
    predictions = pool.map(process_pred, worker_args)
  dev_predictions = [pred[0] for pred in predictions]
  test_predictions = [pred[1] for pred in predictions]
  return dev_predictions, test_predictions

def eval_file(test_preds):
  evaluator = CorefEvaluator()
  for doc_key, document in test_preds.items():
    gold_clusters = document["clusters"]
    predicted_clusters = document["predicted_clusters"]
    (predicted_clusters, gold_clusters,
     mention_to_predicted, mention_to_gold) = mention_maps(predicted_clusters,
                                                             gold_clusters)
    evaluator.update(predicted_clusters, gold_clusters,
                     mention_to_predicted, mention_to_gold)
  return evaluator.get_f1()

def simulate_stopping(split_size, dev_keys, dev_preds, test_f1s, num_samples, gold=None):
  logging.info(f"Starting size {split_size}...")
  # Prepare pool
  worker_args_list = []
  for sample in range(num_samples):
    dev_doc_keys = random.sample(dev_keys, split_size)
    epoch_preds = [{key: dev_pred[key] for key in dev_doc_keys} for dev_pred in dev_preds]
    worker_args_list.append((epoch_preds, test_f1s, sample))
  with Pool(processes=20) as pool:
    sample_metrics = pool.map(simulate_exp, worker_args_list)
  mean = np.mean([m[1] for m in sample_metrics])
  stdev = np.std([m[1] for m in sample_metrics])
  if gold == None:
    matches = sample_metrics[0][2]
  else:
    matches = len([m for m in sample_metrics if m[2] == gold])
  return (mean, stdev, matches)

def simulate_exp(args_list):
  """args_list: (epoch_preds, test_f1s, sample)
  """
  (epoch_preds, test_f1s, sample) = args_list
  # logging.info(f"Starting exp {sample} on {os.getpid()}")
  best_epoch = -1
  best_f1 = -1
  patience = 10
  for epoch, epoch_set in enumerate(epoch_preds):
    dev_f1 = eval_file(epoch_set)
    if dev_f1 > best_f1:
      best_f1 = dev_f1
      best_epoch = epoch
    if epoch - best_epoch >= patience:
      break

  dev_f1 = best_f1
  epoch = best_epoch
  test_f1 = test_f1s[epoch]
  logging.info(f"Stopped at {best_epoch} with {dev_f1} and {test_f1}")
  # logging.info(f"Finish exp {sample}")
  return (dev_f1, test_f1, epoch)

if __name__ == "__main__":
  dev_set = sys.argv[1]
  test_set = sys.argv[2]
  all_checkpoints = sys.argv[4:]
  dev_keys = [json.loads(s)["doc_key"] for s in open(dev_set, 'r')]
  test_keys = [json.loads(s)["doc_key"] for s in open(test_set, 'r')]
  logging.info(f"Testing on {all_checkpoints}")
  for checkpoint_path in all_checkpoints:
    random.seed(SEED)
    pred_files = os.listdir(checkpoint_path)
    preds = [p for p in pred_files if "preds_" in p]
    logging.info(f"{checkpoint_path}: Starting execution")
    dev_preds, test_preds = process_preds(checkpoint_path, preds, dev_keys, test_keys)
    logging.info(f"{checkpoint_path}: Read files")
    with Pool(processes=20) as pool:
      test_f1s = pool.map(eval_file, test_preds)
    logging.info(f"{checkpoint_path}: Done with precomputing testf1s")
    mean, variance, gold = simulate_stopping(500, dev_keys, dev_preds, test_f1s, 1) # Do the full one first
    logging.info(f"{checkpoint_path}: RESULT (mean/std/match): 500: {mean:.5f} {variance:.5f} {gold}")
    splits = [1, 5, 10, 15, 25, 50, 100, 150, 250]
    for split_size in splits:
      mean, variance, matches = simulate_stopping(split_size, dev_keys, dev_preds, test_f1s, NUM_SAMPLES, gold=gold)
      logging.info(f"{checkpoint_path}: RESULT (mean/std/match): {split_size}: {mean:.5f} {variance:.5f} {matches}")

