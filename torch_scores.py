import logging
import os
import numpy as np
import torch

from encoder import Encoder
import util
from util import load_from_pretrained_or_random as load_from

torch.manual_seed(0)

class GenreEmbedder(torch.nn.Module):
  def __init__(self, config, scorer_vars={}):
    super(GenreEmbedder, self).__init__()
    genre_emb_size = [len(config["genres"]), config["genre"]["genre_emb_size"]]
    genre_emb = load_from(scorer_vars, "genre_embeddings", genre_emb_size)
    self.genre_emb = torch.nn.Parameter(data=genre_emb)
    self.genres = {g: i for i, g in enumerate(config["genres"])}
    self.device = config["device"]

  def forward(self, genre_string):
    genre = torch.tensor(self.genres.get(genre_string, 0), device=self.device)
    return torch.index_select(self.genre_emb, 0, genre)


class FFNN(util.FFNN):
  # Thin wrapper for tf parameter initialization
  def __init__(self, config, scorer_vars, ffnn_path, layer_0_dims):
    super(FFNN, self).__init__(layer_0_dims[0], layer_0_dims[1], 1,
                               config["dropout"], output_dim=1)
    hidden_weights_0 = load_from(scorer_vars,
                                 "{}/hidden_weights_0".format(ffnn_path),
                                 layer_0_dims)
    hidden_bias_0 = load_from(scorer_vars,
                              "{}/hidden_bias_0".format(ffnn_path),
                              [layer_0_dims[1]])
    output_weights = load_from(scorer_vars,
                               "{}/output_weights".format(ffnn_path),
                               [layer_0_dims[1]])
    output_bias = load_from(scorer_vars,
                            "{}/output_bias".format(ffnn_path),
                            [])
    self.layers[0].weight.data = hidden_weights_0.t()
    self.layers[0].bias.data = hidden_bias_0
    self.projection.weight.data = output_weights.t()
    self.projection.bias.data = output_bias


class Projection(torch.nn.Module):
  def __init__(self, scorer_vars, ffnn_path, output_dims):
    super(Projection, self).__init__()
    output_weights = load_from(scorer_vars,
                               "{}/output_weights".format(ffnn_path),
                               [output_dims[0]])
    output_bias = load_from(scorer_vars,
                            "{}/output_bias".format(ffnn_path),
                            [])
    self.output_layer = torch.nn.Linear(output_dims[0], output_dims[1])
    self.output_layer.weight.data = output_weights.t().unsqueeze(0)
    self.output_layer.bias.data = output_bias.unsqueeze(0)

  def forward(self, x):
    return self.output_layer(x)


class SpanScorer(torch.nn.Module):
  """ This is a span scorer, scores a single span embedding"""
  def __init__(self, config, scorer_vars={}):
    super(SpanScorer, self).__init__()
    self.config = config
    token_emb_size = config["token_emb_size"]
    feature_size = config["spans"]["feature_size"]
    mention_layer_input_dim = token_emb_size * 3 + feature_size
    mention_layer_output_dim = config["spans"]["output_size"]
    mention_layer_dims = (mention_layer_input_dim, mention_layer_output_dim)
    mention_attn_dims = (token_emb_size, 1)
    width_layer_dims = (feature_size, mention_layer_output_dim)
    span_width_dims = (config["spans"]["span_width_buckets"], feature_size)

    span_width_embeddings = load_from(scorer_vars, "span_width_embeddings", span_width_dims)
    span_width_prior_embeddings = load_from(scorer_vars,
                                            "span_width_prior_embeddings",
                                            span_width_dims)

    self.span_width_embeddings = torch.nn.Parameter(data=span_width_embeddings)
    self.span_width_prior_embeddings = torch.nn.Parameter(data=span_width_prior_embeddings)
    self.attnProjection = Projection(scorer_vars, "mention_word_attn", mention_attn_dims)
    self.spanFFNN = FFNN(self.config, scorer_vars, "mention_scores", mention_layer_dims)
    self.widthFFNN = FFNN(self.config, scorer_vars, "width_scores", width_layer_dims)
    self.max_span_width = config["max_span_width"]
    self.top_span_ratio = config["top_span_ratio"]
    self.device = config["device"]

  def forward(self, starts, ends, embs, use_gold_spans=False):
    incr_doc_embs = embs.squeeze(0)
    span_start_embs = torch.index_select(embs, 1, starts) # [num_spans, emb_size]
    span_end_embs = torch.index_select(embs, 1, ends)

    span_width_index = torch.clamp((ends - starts), max=29).long()
    span_width_embs = torch.index_select(self.span_width_embeddings, 0, span_width_index).unsqueeze(0) # [1, num_spans, 20]
    word_attn = self.attnProjection(incr_doc_embs)
    num_spans = span_start_embs.shape[1]
    num_words = incr_doc_embs.shape[0]
    doc_range = torch.arange(num_words, device=self.device).unsqueeze(0).repeat([num_spans, 1])
    # create mask
    mention_mask = ((doc_range >= starts.unsqueeze(1)) & (doc_range <= ends.unsqueeze(1)))  # [num_spans, num_words]
    mention_word_attn = torch.softmax(torch.log(mention_mask.float()) + word_attn.view(1, -1), dim=1)
    span_head_embs = mention_word_attn.mm(incr_doc_embs).unsqueeze(0) # [1, num_spans, 1024]
    # [1, num_spans, emb_size]
    span_embs = torch.cat([span_start_embs, span_end_embs, span_width_embs, span_head_embs], dim=2)
    # Get span scores
    span_scores = self.spanFFNN(span_embs)
    width_scores = self.widthFFNN(self.span_width_prior_embeddings.unsqueeze(0))
    width_scores = torch.index_select(width_scores, 1, span_width_index)
    total_scores = span_scores + width_scores
    total_scores = total_scores.squeeze(0)
    if use_gold_spans:
      return (span_embs.squeeze(0), total_scores, None)
    else:
      # Assume batch size 1
      k = int(num_words * self.top_span_ratio)
      top_scores, top_k_indices = torch.topk(total_scores, k=k)
      top_span_embs = torch.index_select(span_embs, 1, top_k_indices).squeeze(0)
      return (top_span_embs, top_scores, top_k_indices)


class ScoringModule(torch.nn.Module):
  """ This is the slow, pairwise scorer"""
  def __init__(self, config, scorer_vars={}):
    super(ScoringModule, self).__init__()
    self.config = config

    feature_size = config["spans"]["feature_size"]
    output_size = config["spans"]["output_size"]
    token_emb_size = config["token_emb_size"]
    span_size = token_emb_size * 3 + feature_size
    same_speaker_emb_dims = (config["pairwise"]["speaker_buckets"], feature_size)
    antecedent_distance_dims = (config["pairwise"]["antecedent_distance_buckets"], feature_size)
    segment_distance_dims = (config["pairwise"]["segment_buckets"], feature_size)
    slow_scorer_dims = (span_size * 3 + feature_size * 4, output_size)

    same_speaker_emb = load_from(scorer_vars, "coref_layer/same_speaker_emb", same_speaker_emb_dims)
    antecedent_distance_emb = load_from(scorer_vars, "coref_layer/antecedent_distance_emb", antecedent_distance_dims)
    segment_distance_emb = load_from(scorer_vars, "coref_layer/segment_distance/segment_distance_embeddings", segment_distance_dims)

    self.same_speaker_emb = torch.nn.Parameter(data=same_speaker_emb)
    self.antecedent_distance_emb = torch.nn.Parameter(data=antecedent_distance_emb)
    self.segment_distance_emb = torch.nn.Parameter(data=segment_distance_emb)
    self.slow_antecedent_scores = FFNN(self.config, scorer_vars,
                                       "coref_layer/slow_antecedent_scores", slow_scorer_dims)
    self.device = config["device"]

  def forward(self, span_emb, cluster_embs, offsets, genre_emb):
    """ Everything except offsets is already on device
    since it was computed elsewhere"""
    span_emb = span_emb.unsqueeze(0)
    cluster_ids = torch.arange(len(cluster_embs), device=self.device).unsqueeze(0)
    # Need to do random permutation here to ensure symmetric function.

    cluster_embs = torch.stack(cluster_embs, 0).unsqueeze(0)
    offsets = offsets.unsqueeze(0)
    speakers = torch.zeros_like(cluster_ids).squeeze(0)
    genre_emb = genre_emb.squeeze(0)
    # We could use segment features...
    segment_distance = torch.zeros_like(cluster_ids)
    same_speaker = speakers.unsqueeze(0)
    return self.get_slow_antecedent_scores(
      span_emb,
      cluster_ids,
      cluster_embs,
      offsets,
      genre_emb,
      segment_distance,
      same_speaker)

  def bucket_distance(self, distances):
    """
    Places the given values (designed for distances) into 10 semi-logscale buckets:
    [0, 1, 2, 3, 4, 5-7, 8-15, 16-31, 32-63, 64+].
    """
    logspace_idx = torch.floor(torch.log(distances.float().abs())/np.log(2)).int() + 3
    use_identity = (distances <= 4).int()
    combined_idx = use_identity * distances + (1 - use_identity) * logspace_idx
    # return torch.zeros_like(distances).long()
    return torch.clamp(combined_idx, 0, 9).long()

  def get_slow_antecedent_scores(
      self,
      top_span_emb,
      top_antecedents,
      top_antecedent_emb,
      top_antecedent_offsets,
      genre_emb,
      segment_distance,
      same_speaker):
    """
    This function has a bunch of issues with batching, it can only do batch size 1
    due to shape mismatches and pytorch is difficult to work with on this unless
    we include batched_index_select
    """
    k = top_span_emb.size(0) # but it is 1
    c = top_antecedents.size(1)

    feature_emb_list = [] # use all features by default

    # This section is hardcoded to batch size 1, batching is not practical...
    # Should not actually have speakers, since that mathematically breaks entity reps
    speaker_pair_emb = torch.index_select(self.same_speaker_emb, 0, same_speaker[0])
    speaker_pair_emb = torch.zeros_like(speaker_pair_emb)
    feature_emb_list.append(speaker_pair_emb) # [c, emb]

    tiled_genre_emb = genre_emb.unsqueeze(0).repeat([c, 1]) # [c, emb]
    feature_emb_list.append(tiled_genre_emb)

    antecedent_distance_buckets = self.bucket_distance(top_antecedent_offsets.int())
    antecedent_distance_emb = torch.index_select(self.antecedent_distance_emb, 0,
                                                 antecedent_distance_buckets[0])
    feature_emb_list.append(antecedent_distance_emb) # [c, emb]
    # Assume same segment? or ignore entirely.
    segment_distance_emb = torch.index_select(self.segment_distance_emb, 0,
                                              segment_distance[0])
    feature_emb_list.append(segment_distance_emb) # [c, emb]
    feature_emb = torch.cat(feature_emb_list, 1).unsqueeze(0) # this depends on batch size 1

    target_emb = top_span_emb.unsqueeze(1) # [k, 1, emb]
    similarity_emb = top_antecedent_emb * target_emb # [k, c, emb]
    target_emb = target_emb.repeat([1, c, 1])
    pair_emb = torch.cat([target_emb, top_antecedent_emb, similarity_emb, feature_emb], 2)
    slow_antecedent_scores = self.slow_antecedent_scores(pair_emb)
    return slow_antecedent_scores

def create_from_scratch(config=None):
  if config is None:
    config = util.initialize_from_env()
  device = torch.device('cuda') if torch.cuda.is_available() else "cpu"
  config["device"] = device

  # Initialization
  encoder = Encoder(config).to(device=device)

  # If ckpt_dir does not exist or if we want to reset weights
  if config["reset_weights"] or "ckpt_dir" not in config:
    logging.info(f"Resetting all weights with normal or xavier_uniform (seed {torch.initial_seed()}) because reset_weights=True or ckpt_dir not found")
    torch_scorer_vars = {}
  else:
    init_path = os.path.join(config["ckpt_dir"], "torch_scorer_vars.bin")
    try:
      torch_scorer_vars = torch.load(init_path)
      logging.info(f"Loaded model from {init_path}")
    except FileNotFoundError:
      torch_scorer_vars = {}
      logging.info(f"Failed to load model from {init_path}. Resetting instead")
  genre_embedder = GenreEmbedder(config, scorer_vars=torch_scorer_vars).to(device=device)
  span_scorer = SpanScorer(config, scorer_vars=torch_scorer_vars).to(device=device)
  scoring_module = ScoringModule(config, scorer_vars=torch_scorer_vars).to(device=device)
  logging.info("Putting Encoder, GenreEmbedder, SpanScorer, and ScoringModule all on {}".format(device))

  return encoder, genre_embedder, span_scorer, scoring_module, device


if __name__ == "__main__":
  create_from_scratch()
