from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import collections
import sys
import numpy as np
import errno
import logging
import json
import _jsonnet
import torch
import random
from cluster import Span

def initialize_from_env(eval_test=False, use_overrides=True):
  if "GPU" in os.environ:
    set_gpus(int(os.environ["GPU"]))

  name = sys.argv[1]
  overrides = {}
  if len(sys.argv) > 2 and use_overrides:
    for item in sys.argv[2:]:
      key, value = item.split("=", 1)
      try:
        overrides[key] = json.loads(value)
      except:
        overrides[key] = value

  config = json.loads(_jsonnet.evaluate_file("experiments.jsonnet"))[name]

  # Put everything in override
  config.update(overrides)

  mkdirs(config["log_dir"])
  logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt = "%Y-%m-%d %H:%M:%S",
    handlers=[
        logging.FileHandler(config["log_dir"] + "/out.log"),
        logging.StreamHandler()
    ]
  )
  logging.getLogger("transformers").setLevel(logging.WARNING)
  logging.info("Running experiment: {}".format(name))
  logging.info(json.dumps(config, indent=2))
  device = torch.device('cuda') if torch.cuda.is_available() else "cpu"
  config["device"] = device
  if "load_path" not in config:
    config["load_path"] = config["log_path"]
  return config


def mkdirs(path):
  try:
    os.makedirs(path)
  except OSError as exception:
    if exception.errno != errno.EEXIST:
      raise
  return path


def load_params(module, log_path, key):
  try:
    checkpoint = torch.load(log_path, map_location="cpu")
    logging.info(f"Found checkpoint at {log_path}, loading instead.")
  except:
    logging.info(f"Checkpoint not found at {log_path}")
    return
  try:
    missing, unexpected = module.load_state_dict(checkpoint[key], strict=False)
    if missing or unexpected:
      print(f"Did not find (using defaults):{str(missing)[:100]}...\n\n" +
            f"Unexpected params (ignoring): {unexpected}")
  except Exception as e1:
    try:
      module.load_state_dict(checkpoint[key])
      logging.info(f"Module cannot load with keyword strict=False. Loading with its default.")
    except Exception as e2:
      logging.info(f"Unable to load checkpoint for {key}: {e1}, {e2}")
      return


def load_data(path, num_examples=None):
  if path is None or not path:
    return []
  def load_line(line):
    example = json.loads(line)
    # Need to make antecedent dict
    clusters = [sorted(cluster) for cluster in example["clusters"]]
    antecedent_map = {}
    for cluster in clusters:
      antecedent_map[tuple(cluster[0])] = "0"
      for span_idx in range(1, len(cluster)):
        antecedent_map[tuple(cluster[span_idx])] = [tuple(span) for span in cluster[:span_idx]]
    example["antecedent_map"] = antecedent_map
    return example
  with open(path) as f:
    data = [load_line(l) for l in f.readlines()]
    if num_examples is not None:
      data = data[:num_examples]
    logging.info("Loaded {} examples.".format(len(data)))
    return data


def flatten(l):
  return [item for sublist in l for item in sublist]

def safe_add(tensor1, tensor2):
  # None is the additive identity and the result can be backpropped
  if tensor1 is None:
    return tensor2
  if tensor2 is None:
    return tensor1
  else:
    return tensor1 + tensor2


# For memory debugging:

# from pytorch_memlab import MemReporter

# def track(name=None):
#   reporter = MemReporter(name)
#   reporter.report()

# def num_obj():
#   reporter = MemReporter()
#   reporter.collect_tensor()
#   reporter.get_stats()
#   return reporter.device_tensor_stat[torch.device("cuda:0")]


def get_cuda_memory_allocated():
  GB = 1073741824 # 1024 ** 3
  if torch.cuda.is_available():
    memory = torch.cuda.memory_allocated() / GB
  else:
    memory = 0.0
  return memory

def get_segment_iter(document):
  if "mask_windows" in document and "segment_window" in document:
    index_masks = [[idx for idx, x in enumerate(mask_window) if x >= 0]
                   for mask_window in document["mask_windows"]]
    seglens = [len(mask) for mask in index_masks]
    segments = zip(document["segment_window"], index_masks, seglens)
  else:
    seglens = [len(sentence) for sentence in document["sentences"]]
    segments = zip(document["sentences"], [None for _ in document["sentences"]], seglens)
  return enumerate(segments)


def get_sentence_iter(sentences, segment_map, data_loader, seg_offset, genre, cluster_fn):
  lower = 0
  min_sent = segment_map[0]
  max_sent = segment_map[-1]
  num_sents = max_sent - min_sent
  for curr_sent in range(min_sent, max_sent):
    upper = lower + sum([idx == curr_sent for idx in segment_map])
    sentence = sentences[lower:upper]
    spans = [cluster_fn(Span(emb,
                          start.item() + seg_offset,
                          end.item() + seg_offset,
                          lower + seg_offset,
                          sentence,
                          score))
             for emb, start, end, score in data_loader
             if ((lower <= start and start < upper) and
                 (lower <= end and end < upper))]
    spans = sorted(spans)
    lower = upper
    yield (curr_sent, genre, spans)


def mention_maps(predicted_clusters, gold_clusters):
  gold_clusters = [tuple(tuple(m) for m in gc) for gc in gold_clusters]
  predicted_clusters = [tuple(tuple(m) for m in pc) for pc in predicted_clusters]
  mention_to_gold = {}
  for gc in gold_clusters:
    for mention in gc:
      mention_to_gold[mention] = gc
  mention_to_predicted = {}
  for pc in predicted_clusters:
    for mention in pc:
      mention_to_predicted[mention] = pc
  return predicted_clusters, gold_clusters, mention_to_predicted, mention_to_gold

def segment_map(sentences):
  # Used in eval_all.py
  offset = 0
  m = {}
  for i, segment in enumerate(sentences):
    for tok_idx, _ in enumerate(segment):
      m[offset + tok_idx] = i
    offset += len(segment)
  return m

def make_evict_function(params):
  if type(params) != dict or "name" not in params:
    return lambda cluster, idx: False

  if params["name"] == "singletons":
    return lambda cluster, idx: (len(cluster) == 1 and
                                 idx - cluster.start > params["distance"])
  elif params["name"] == "trunc_linscale":
    return lambda cluster, idx: (len(cluster) == 1 and idx - cluster.start > params["distance"]
                                 or idx - cluster.start > 2 * params["distance"])
  else:
    return lambda cluster, idx: False

class FFNN(torch.nn.Module):
  def __init__(self, input_dim, hidden_dim, num_layers, dropout, output_dim=None):
    super(FFNN, self).__init__()
    _dropout = [torch.nn.Dropout(p=dropout) for _ in range(num_layers)]
    # hidden_dims is a list, num_layers > 0
    # [(input, hidden) + (hidden, hidden) ....]
    layer_dims = [(input_dim, hidden_dim)] + ([(hidden_dim, hidden_dim)] * (num_layers - 1))
    _layers = [torch.nn.Linear(i, o) for i, o in layer_dims]
    if output_dim is not None:
      self.projection = torch.nn.Linear(hidden_dim, output_dim)
    else:
      self.projection = None
    self.layers = torch.nn.ModuleList(_layers)
    self.dropout = torch.nn.ModuleList(_dropout)

  def forward(self, x):
    output = x
    for layer, dropout in zip(self.layers, self.dropout):
      output = dropout(torch.relu(layer(output)))
    if self.projection is not None:
      output = self.projection(output)
    return output

def random_tensor(sizes):
  if len(sizes) >= 2:
    return torch.nn.init.xavier_uniform_(
      torch.empty(sizes))
  else:
    return torch.nn.init.normal_(torch.empty(sizes))

def gen_subepoch_iter(data, size):
  num_pieces = int((len(data) - 1)/size) + 1
  return [data[i*size:(i+1)*size] for i in range(num_pieces)]

def load_from_pretrained_or_random(scorer_vars, name, sizes):
  return scorer_vars.get(name, random_tensor(sizes))

def set_seed(config):
  # There is still nondeterminism somewhere
  logging.info(f"Setting seed to {config['seed']}")
  random.seed(config['seed'])
  np.random.seed(config['seed'])
  torch.manual_seed(config['seed'])
  torch.cuda.manual_seed_all(config['seed'])
