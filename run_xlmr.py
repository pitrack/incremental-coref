"""
Run file assuming xlm-r tokenizer. Part of LOME project.
"""

import logging
import json
import sys
import torch

from conversion_scripts.minimize_json import get_document
from incremental import Incremental
from inference import Predictor
from transformers import XLMRobertaTokenizer
import util

logging.basicConfig(level=logging.INFO)

def setup_model(config):
  config = util.initialize_from_env(use_overrides=False)
  incremental_model = Incremental(config)
  util.load_params(incremental_model, config["log_path"], "model")
  logging.info(f"Updating threshold to {config['threshold']}")
  incremental_model.set_threshold(config["threshold"])

  predictor = Predictor(incremental_model, [], config["singleton_eval"])
  return predictor

def predict(predictor, data):
  data["antecedent_map"] = {}  # Placeholder
  predictor.data = [data]  # We only have one example but it expects list
  with torch.no_grad():
    predictor.evaluate(perf=False)
  predictions = predictor.predictions
  return predictions

def minimize_data(json_blob):
  tokenizer = XLMRobertaTokenizer.from_pretrained('xlm-roberta-large')
  text = []
  for sentence in json_blob["sentences"]:
    text.extend([[word] for word in sentence])
    text.append([])
  minimized_json = get_document((json_blob["doc_key"], text, json_blob["clusters"]),
                                tokenizer, None, 512)
  return minimized_json

# Concrete-related wrappers

def read_from_concrete(concrete_input):
  """We can assume there's only one input file

  Returns a thin dict with text, doc key, and clusters
  """
  (data, comm) = next(reader.make_data_iter(concrete_input, None))
  data["doc_key"] = data["doc_id"] # same thing, doc key is used later
  return ((data, comm), {
    "doc_key": data["doc_id"],
    "sentences": data["sentences"],
    "clusters": [[list(span)] for span in data["mentions"]]
  })

def write_to_concrete(line_dict, examples_dict, concrete_output):
  augf = AnalyticUUIDGeneratorFactory()
  aug = augf.create()
  metadata = lambda: cmt.AnnotationMetadata(tool="jhu_xlmr_0:incremental_coref_v1",
                                            timestamp=now_timestamp())
  comm, _, _ = writer.convert_to_comm(line_dict, examples_dict, aug, metadata)
  write_communication_to_file(comm, concrete_output)

# Run files

def run(config, json_path, preds_file):
  # Only runs first example. This is raw json and needs to be minimized
  data = [json.loads(line) for line in open(json_path, 'r')][0]

  # Minimization is not idempotent, attempting a second time will malform input
  data = minimize_data(data)

  predictor = setup_model(config)
  predictions = predict(predictor, data)
  preds_output = open(preds_file, 'w+')
  preds_output.write(json.dumps([preds["clusters"] for preds in predictions]))

def run_concrete(config, concrete_input, concrete_output):
  from concrete_wrapper import reader, writer
  from concrete.util.concrete_uuid import AnalyticUUIDGeneratorFactory
  from concrete.util import write_communication_to_file, now_timestamp
  import concrete.metadata.ttypes as cmt
  full_data, data = read_from_concrete(concrete_input) # Return a dict
  data = minimize_data(data)
  predictor = setup_model(config)
  predictions = predict(predictor, data)[0] # Only care about last one
  data["predicted_clusters"] = json.loads(json.dumps(predictions["clusters"]))
  examples_dict = {data["doc_key"]: full_data}
  write_to_concrete(data, examples_dict, concrete_output)

if __name__ == "__main__":
  config = sys.argv[1]
  in_path = sys.argv[2]
  out_path = sys.argv[3]
  # In LOME, we use input/output formatted under the concrete data scheme
  # If in_path/out_path is plaintext (json), concrete should be False
  concrete = sys.argv[4]

  if concrete:
    run_concrete(config, in_path, out_path)
  else:
    run(config, in_path, out_path)
