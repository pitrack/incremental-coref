import torch
import time
import json
import logging
import conll
from tqdm import tqdm
from collections import defaultdict
from metrics import CorefEvaluator
from eval_all import update_evaluators, ALL
from incremental import Incremental
from cluster import ClusterList
import util
import sys

class Predictor():
  def __init__(self, model, data):
    self.model = model
    self.data = data
    self.reset()

  def reset(self):
    self.model.reset_metrics()
    self.evaluators = defaultdict(CorefEvaluator)
    self.predictions = []
    if torch.cuda.is_available():
      torch.cuda.reset_max_memory_allocated()

  def evaluate(self, perf=False, inspect=False, conll_path=None):
    logging.info(f"loss count {self.model.loss_count} and sampled loss count {self.model.sampled_loss_count}")
    self.reset()
    self.predictions = []
    coref_predictions = {} # for conll eval
    self.model = self.model.eval()
    self.perf_data = []
    start_time = time.time()
    eval_iterator = tqdm(enumerate(self.data))
    total_loss = []
    if inspect:
      debug_cluster_lists = []
    for doc_id, document in eval_iterator:
      if inspect and inspect < 500:
        if doc_id != inspect:
          continue
      cluster_list, loss = self.incremental_clustering(document)
      predicted_clusters = cluster_list.get_clusters(condensed=True, print_clusters=False)
      update_evaluators(self.evaluators, document, predicted_clusters, document["clusters"])
      clusters_as_list = cluster_list.as_list()
      self.predictions.append([c for c in clusters_as_list if len(c) > 1])
      coref_predictions[document["doc_key"]] = [c for c in clusters_as_list if len(c) > 1]
      if perf and torch.cuda.is_available():
        max_memory_alloc = torch.cuda.max_memory_allocated() / (1024 ** 3)
        torch.cuda.reset_max_memory_allocated()
        torch.cuda.empty_cache()
        self.perf_data.append((document["subtoken_map"][-1], max_memory_alloc))
      eval_iterator.set_description_str(desc=f"{loss:.4f}")
      total_loss.append(loss)
      if inspect:
        cluster_list.detach_()
        debug_cluster_lists.append(cluster_list)
    avg_loss = sum(total_loss) / len(total_loss)
    logging.info(f"cn: {self.model.correct_new} and wn: {self.model.wrong_new}")
    logging.info(f"ca: {self.model.correct_attach} and wa: {self.model.wrong_attach}")
    end_time = time.time()
    logging.info(f"Evaluation on {len(self.data)} documents [{avg_loss:.3f}] took {end_time - start_time:.1f} seconds")
    eval_dict = [f"{key}: {evaluator.prf_str()}, ({evaluator.get_count()} docs)" for key, evaluator in self.evaluators.items()]
    for line in list(sorted(eval_dict)):
      logging.info(line)
    if perf:
      logging.info(self.perf_data)
    self.model = self.model.train()
    if inspect:
      return debug_cluster_lists
    self.evaluators[ALL].get_full()

    if conll_path is not None:
      subtoken_maps = {document["doc_key"]: document["subtoken_map"]
                       for document in self.data}
      eval_dict = conll.evaluate_conll(conll_path, coref_predictions,
                                       subtoken_maps, True)
      print([evaluator.__dict__ for evaluator in self.evaluators[ALL].evaluators])
  
    return self.evaluators[ALL].get_f1()

  def incremental_clustering(self, document):
    total_clusters = ClusterList()
    loss = 0.0
    start_idx = 0
    segment_iter = util.get_segment_iter(document)
    for seg_id, (segment, mask, seglen) in segment_iter:
      loss += self.model(segment, document, total_clusters, start_idx, mask)
      start_idx += seglen
      self.model.clear_cache(total_clusters, start_idx)
    total_clusters.cpu_()
    return total_clusters, loss

  def write_preds(self, path):
    preds_file = open(path, 'w+')
    for document, preds in zip(self.data, self.predictions):
      document["predicted_clusters"] = preds
      document_write = dict(document)
      del document_write["antecedent_map"]
      preds_file.write(json.dumps(document_write) + "\n")
    preds_file.close()

if __name__ == "__main__":
  config = util.initialize_from_env()
  if len(sys.argv) > 1:
    conll_path = sys.argv[2]
  else:
    conll_path = None
  eval_data = util.load_data(config["eval_path"])
  incremental_model = Incremental(config)
  if config["load_model"] != "false":
    try:
      checkpoint = torch.load(config["log_path"], map_location="cpu")
      logging.info(f"Found old model at {config['log_path']}, loading instead")
      missing, unexpected = incremental_model.load_state_dict(checkpoint["model"])
      if missing or unexpected:
        print(missing, unexpected)
    except Exception as e:
      logging.info(f"Old model not found or failed to load: {e}")
  predictor = Predictor(incremental_model, eval_data)
  with torch.no_grad():
    predictor.evaluate(perf=False, conll_path=conll_path)

  if "preds_file" in config:
    preds_file = config["preds_file"]
  else:
    preds_file = config['log_dir']+'/preds.json'
  predictor.write_preds(preds_file)
  logging.info(f"Wrote preds to {preds_file}")
