import tensorflow as tf
import torch
import sys
# from torch_scores import *

tf_path = sys.argv[1]
init_vars = tf.train.list_variables(tf_path)
torch_scorer_vars = {}
ffnn_vars = ["hidden_bias_0", "hidden_weights_0",
             "output_bias", "output_weights"]
torch_scorer_var_names = (
  [
    "coref_layer/antecedent_distance_emb",
    "coref_layer/same_speaker_emb",
    "coref_layer/segment_distance/segment_distance_embeddings",
    "span_width_prior_embeddings",
    "genre_embeddings",
    "span_width_embeddings",
  ] + [
      "coref_layer/slow_antecedent_scores/{}".format(ffnn_var)
      for ffnn_var in ffnn_vars
  ] +
  ["mention_word_attn/{}".format(ffnn_var) for ffnn_var in ffnn_vars] +
  ["mention_scores/{}".format(ffnn_var) for ffnn_var in ffnn_vars] +
  ["width_scores/{}".format(ffnn_var) for ffnn_var in ffnn_vars])

for name, shape in init_vars:
  if name not in torch_scorer_var_names:
    continue
  print("Loading TF weight {} with shape {}".format(name, shape))
  array = tf.train.load_variable(tf_path, name)
  torch_scorer_vars[name] = torch.from_numpy(array.squeeze())

torch.save(torch_scorer_vars, sys.argv[2])


