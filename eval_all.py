import util
import argparse
import json
import numpy as np
from metrics import CorefEvaluator, Evaluator
from collections import defaultdict
from constants import ONTONOTES_SEGMENT_EVAL, ALL

def read_file(path):
  d = {}
  with open(path, 'r') as f:
    for line in f:
      document_blob = json.loads(line)
      if document_blob["doc_key"] in d:
        document_blob["doc_key"] = document_blob["doc_key"] + "dup"
      d[document_blob["doc_key"]] = document_blob
  return d

def bucket_tokens(subtoken_map):
  num_tokens = subtoken_map[-1]
  if num_tokens <= 128:
    return ("0-128")
  elif num_tokens <= 256:
    return ("128-256")
  elif num_tokens <= 512:
    return ("256-512")
  elif num_tokens <= 768:
    return ("512-768")
  elif num_tokens <= 1152:
    return ("768-1152")
  else:
    return ("_1152+")

def bucket_tokens_by_seg(doc_key):
  # Only works for ontonotes dev
  segment_count = ONTONOTES_SEGMENT_EVAL.get(doc_key, 0)
  if segment_count <= 1:
    return ("0-128")
  elif segment_count <= 2:
    return ("128-256")
  elif segment_count <= 4:
    return ("256-512")
  elif segment_count <= 6:
    return ("512-768")
  elif segment_count <= 9:
    return ("768-1152")
  else:
    return ("_1152+")

def bucket_segments(sentences):
  num_segments = len(sentences)
  if num_segments <= 1:
    return ("1")
  elif num_segments <= 2:
    return ("2")
  elif num_segments <= 3:
    return ("3")
  elif num_segments <= 4:
    return ("4")
  else:
    return (">5")

def update_evaluators(evaluators, document, predicted_clusters, gold_clusters):
  (predicted_clusters, gold_clusters,
   mention_to_predicted, mention_to_gold) = util.mention_maps(predicted_clusters,
                                                              gold_clusters)
  def keyed_update(key):
    evaluators[key].update(predicted_clusters, gold_clusters, mention_to_predicted, mention_to_gold)
  genre = document["doc_key"][:2]
  token_bucket = bucket_tokens(document["subtoken_map"])
  has_speakers = "_speaker_" + str(genre in ["bc", "tc"])
  update_keys = [
    ALL,
    # "_genre_" + genre + "_t_" +  token_bucket,
    # "_genre0" + has_speakers,
    # "_s_" + bucket_segments(document["sentences"]),
    # "_s+t_" + bucket_tokens_by_seg(document["doc_key"]),
    "_t_" + token_bucket,
    # genre,
  ]
  if "language" in document:
    update_keys.append(document["language"])

  update_keys = set(update_keys)
  for key in update_keys:
    keyed_update(key)

def print_cluster(doc, clusters):
  tokens = util.flatten(doc["sentences"])
  for i, cluster in enumerate(clusters):
    if len(cluster) > 1:
      print(f"cluster {i}")
      for span in cluster:
        print(f"{' '.join(tokens[span[0]:span[1] + 1])} [{span[0]}, {span[1] + 1}]")
      print("-" * 80)

def count_crossings(clusters, segment_map):
  # does the "direct" variant
  seam_spans = set()
  relaxed_seam_spans = {}
  for cluster in clusters:
    spans = sorted(cluster, key=lambda x: x[0])
    prev_span = spans[0]
    for i, span in enumerate(spans[1:]):
      if segment_map[prev_span[0]] != segment_map[span[0]]:
        # break
        seam_spans.add((tuple(prev_span), tuple(span)))
        relaxed_seam_spans[tuple(span)] = spans[:i+1] # any antecedent is okay
      prev_span = span
  return seam_spans, relaxed_seam_spans

def seam_evaluation(exp):
  strict = Evaluator(metric=None)
  relaxed = Evaluator(metric=None)
  for key, document in exp.items():
    gold_clusters = document["clusters"]
    predicted_clusters = document["predicted_clusters"]
    # For each cluster, for each new mention, what's the accuracy it linked to the
    # right cluster from earlier (i.e. is its direct/any antecedent included in the gold set?)
    segment_map = util.segment_map(document["sentences"])
    gold_crossings, relaxed_gold = count_crossings(gold_clusters, segment_map)
    predicted_crossings, _ = count_crossings(predicted_clusters, segment_map)
    intersection = gold_crossings & predicted_crossings
    relaxed_intersection = [(ant, span) for ant, span in predicted_crossings
                            if list(ant) in relaxed_gold.get(span, [])]
    gold_seam = len(gold_crossings)
    predicted_seam = len(predicted_crossings)
    intersection_seam = len(intersection)
    relaxed_intersection_seam = len(relaxed_intersection)
    strict.raw_update(intersection_seam, predicted_seam, gold_seam)
    relaxed.raw_update(relaxed_intersection_seam, predicted_seam, gold_seam)
  sp, sr, sf = strict.get_prf()
  rp, rr, rf = relaxed.get_prf()
  print(f"[STRICT] RECALL (accuracy) {sr:.3f}, p: {sp:.3f}, f1: {sf:.3f}")
  print(f"[RELAX] RECALL (accuracy) {rr:.3f}, p: {rp:.3f}, f1: {rf:.3f}")


def calc_spread(cluster):
  # distance = min and max
  # variance = take set of all points and find variance
  cluster = sorted(cluster)
  points = sorted(util.flatten(cluster))
  distance = points[-1] - points[0]
  variance = np.std([(c[0] + c[1]) / 2 for c in cluster])
  size = len(cluster)
  diffs = [cluster[i+1][0] - cluster[i][0] for i in range(size - 1)]
  max_hop = max(diffs)
  mean_hop = np.average(diffs)
  hop_var = np.std(diffs)
  return (distance, variance, max_hop, mean_hop, hop_var, size)

def renumber(clusters, sentences):
  # assumes [cls] and [sep] for each sentence
  segment_map = util.segment_map(sentences)
  def fix_span(span):
    num_fillers = 1 + 2 * segment_map[span[0]]
    return [span[0] - num_fillers,
            span[1] - num_fillers]
  def fix_cluster(cluster):
    return [fix_span(span) for span in cluster]
  return [fix_cluster(cluster) for cluster in clusters]

def distance_eval(exp):
  gold_spread = []
  predicted_spread = []
  def aggregate(distances, variances, max_hops, mean_hops, hop_vars, sizes):
    ret_val = {
      "avg_dist": np.average(distances),
      "max_dist": np.max(distances),
      "avg_var": np.average(variances),
      "max_hop": np.max(max_hops),
      "mean_hop": np.average(mean_hops),
      "avg_hop_var": np.average(hop_vars),
      "avg_size": np.average(sizes)
    }
    return ret_val

  for _, document in exp.items():
    gold_clusters = renumber(document["clusters"], document["sentences"]) # [CLS] and [SEP] get in the way
    predicted_clusters = renumber(document["predicted_clusters"], document["sentences"])
    gold_spread.extend([calc_spread(cluster) for cluster in gold_clusters if len(cluster) > 1])
    predicted_spread.extend([calc_spread(cluster) for cluster in predicted_clusters if len(cluster) > 1])
  gold_stats = aggregate(*list(zip(*gold_spread)))
  predicted_stats = aggregate(*list(zip(*predicted_spread)))
  print({key: f"{val:.2f}" for key, val in gold_stats.items()})
  print({key: f"{val:.2f}" for key, val in predicted_stats.items()})

def evaluate_exp(exp, simple=False):
  evaluators = defaultdict(CorefEvaluator)
  for key, document in exp.items():
    gold_clusters = document["clusters"]
    predicted_clusters = document["predicted_clusters"]
    update_evaluators(evaluators, document, predicted_clusters, gold_clusters)
  if simple:
    print(f"p/r/avg f1: {evaluators[ALL].prf_str()}, ({evaluators[ALL].get_count()} docs) [{evaluators[ALL].evaluators[3].get_f1()} ment f1]")
  else:
    eval_dict = {key: f"{key}: {evaluator.prf_str()}, ({evaluator.get_count()} docs)"
                 for key, evaluator in evaluators.items()}
    print("\n".join(list(sorted(eval_dict.values()))))
    print("\nmetric\tprec.\trec.\tf1\n"+"\n".join(evaluators[ALL].get_full()))
  return evaluators[ALL].get_f1()

if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument("-f", "--files", required=True, nargs="*", type=str, help="List of preds.json files")
  parser.add_argument("-s", "--single", action="store_true",
                      help="If true, merges all predictions into single file. The resulting scores will be different from simply averaging")
  parser.add_argument("-a", "--all_metrics", action="store_true", help="Print intermediate metrics on subsplits.")
  parser.add_argument("-m", "--mean", action="store_true", help="Report mean and std of all runs")

  all_scores = []
  args = parser.parse_args()
  if args.single:
    all_preds = {}
    for pred_file in args.files:
      preds = read_file(pred_file)
      all_preds.update(preds)
    evaluate_exp(all_preds, simple=not args.all_metrics)
  else:
    for pred_file in args.files:
      print(pred_file, end=", ")
      preds = read_file(pred_file)
      f1 = evaluate_exp(preds, simple=not args.all_metrics)
      if args.mean:
        all_scores.append(f1)
  if args.mean:
    print(all_scores)
    print(f"Mean: {100 * np.mean(all_scores) :.4f} with std: {100 * np.std(all_scores):.4f}")
