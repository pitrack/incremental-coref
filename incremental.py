import torch
from torch_scores import create_from_scratch
import util
from cluster import Cluster
from merge_functions import MERGE_NAMES
import random

class Incremental(torch.nn.Module):
  def __init__(self, config):
    super(Incremental, self).__init__()
    (self.encoder, self.genre_embedder,
     self.span_scorer, self.pair_scorer,
     self.device) = create_from_scratch(config)
    self.threshold = torch.nn.Parameter(data=torch.Tensor([config["threshold"]]),
                                        requires_grad=False)
    self.max_span_width = config["max_span_width"]
    self.memory_limit = config["memory_limit"]
    self.negative_sample_rate = config["negative_sample_rate"]
    self.use_gold_spans = config["use_gold_spans"]
    self.teacher_forcing_rate = config["teacher_forcing_rate"]
    self.merge_fn = self.get_merge_fn(config["method"], config["dropout"])
    self.cluster = lambda x: Cluster(x, self.merge_fn)
    self.finetune = config["finetune"]
    # Debugging information
    self.correct_new = 0
    self.correct_attach = 0
    self.wrong_new = 0
    self.wrong_attach = 0
    self.loss_count = 0
    self.sampled_loss_count = 0
    # important for moving threshold to gpu
    self.to(self.device) # This line might be a noop?
    self.evict_fn = util.make_evict_function(config["evict_fn"])

  def reset_metrics(self):
    self.correct_new = 0
    self.correct_attach = 0
    self.wrong_new = 0
    self.wrong_attach = 0
    self.loss_count = 0
    self.sampled_loss_count = 0

  def get_candidate_offsets(self, start_idx, num_words, flattened_sentence_indices):
    """It should also be on gpu?
    """
    flattened_sentence_indices = torch.tensor(flattened_sentence_indices[start_idx:start_idx + num_words], device=self.device) # [num_words]
    candidate_starts = torch.arange(num_words, device=self.device).unsqueeze(1).repeat([1, self.max_span_width]) # [num_words * max_span_width]
    candidate_ends = (candidate_starts + torch.arange(self.max_span_width, device=self.device).unsqueeze(0)).reshape(-1) # [num_words * max_span_width]
    candidate_starts = candidate_starts.reshape(-1) # need to reshape after broadcasted add
    candidate_start_sentence_indices = torch.index_select(flattened_sentence_indices, 0, candidate_starts) # [num_words, max_span_width]
    candidate_end_sentence_indices = torch.index_select(flattened_sentence_indices, 0,
                                                        torch.min(candidate_ends, torch.tensor(num_words - 1, device=self.device))) # [num_words * max_span_width]
    flattened_candidate_mask = ((candidate_ends < num_words) &
                                (candidate_start_sentence_indices == candidate_end_sentence_indices)) # [num_words * max_span_width]
    candidate_starts = torch.masked_select(candidate_starts, flattened_candidate_mask) # [num_candidates]
    candidate_ends = torch.masked_select(candidate_ends, flattened_candidate_mask) # [num_candidates]
    # Might be useful for gold-span setting?
    # candidate_sentence_indices = torch.masked_select(candidate_start_sentence_indices,
    #                                                  flattened_candidate_mask) # [num_candidates]
    return candidate_starts, candidate_ends

  def read_given_spans(self, spans, start_idx, num_words):
    valid_spans = [span for span in spans
                   if span[0] >= start_idx and span[1] < start_idx + num_words]
    starts = torch.tensor([span[0] - start_idx for span in valid_spans], device=self.device)
    ends = torch.tensor([span[1] - start_idx for span in valid_spans], device=self.device)
    return starts, ends

  def get_top_spans(self, sentence_map, doc_embs, start_idx, given_spans=None):
    num_words = doc_embs.shape[1]
    if not self.use_gold_spans:
      starts, ends = self.get_candidate_offsets(start_idx, num_words, sentence_map)
      top_span_embs, scores, top_k_indices = self.span_scorer(starts, ends, doc_embs)
      starts = torch.index_select(starts, 0, top_k_indices)
      ends = torch.index_select(ends, 0, top_k_indices)
    else:
      starts, ends = self.read_given_spans(given_spans, start_idx, num_words)
      if starts.shape[0] == 0:
        return []
      top_span_embs, scores, _ = self.span_scorer(starts, ends, doc_embs, use_gold_spans=True)
    return list(zip(top_span_embs, starts.cpu(), ends.cpu(), scores))

  def resolve_local(self, clusters, sent_gen, gold_map, train=False):
    total_loss = None
    cpu_loss = 0
    num_spans = 0
    for sent_idx, genre, spans in sent_gen:
      for span in spans:
        gold_span = gold_map.get((span.start, span.end), None)
        gold_cluster_id = clusters.get_cluster_id(gold_span) # in range(0, num_clusters + 1)
        if len(clusters) == 0:
          clusters.append(self.cluster(span))
        else:
          cluster_embs = [c.emb for c in clusters]
          antecedent_starts = torch.tensor([(c.start/2.0 + c.end/2.0) for c in clusters], device=self.device)
          computed_offsets = (span.start + span.end)/2.0 - antecedent_starts  # type: int - tensor -> tensor
          pairwise_scores = self.pair_scorer(span.emb, cluster_embs, computed_offsets, genre)  # [1,1] score
          scores = pairwise_scores[0] + span.score
          p_cluster = -1 * torch.log_softmax(torch.cat([self.threshold, scores]), dim=0) # [num_clusters + 1]
          p_score, best_cluster_idx = torch.min(p_cluster, dim=0)
          loss = p_cluster[gold_cluster_id]
          # Update the ClusterList structure
          if not train or (random.uniform(0, 1) > self.teacher_forcing_rate):
            if best_cluster_idx == 0:
              clusters.append(self.cluster(span))
            else:
              clusters.merge_span(int(best_cluster_idx.item()), span, score=p_score)
          else:
            if gold_cluster_id == 0:
              clusters.append(self.cluster(span))
            else:
              clusters.merge_span(gold_cluster_id, span, score=p_score)
          # Update loss or metrics
          if gold_cluster_id != 0:
            total_loss = loss if total_loss is None else total_loss + loss
            self.loss_count += 1
            num_spans += 1
          elif random.uniform(0, 1) < self.negative_sample_rate: # hyperparameter
            total_loss = loss if total_loss is None else total_loss + loss
            self.sampled_loss_count += 1
            num_spans += 1
          # Stats on updates
          if not train:
            self.compute_attach_stats(best_cluster_idx, gold_cluster_id)
        if total_loss is not None and (not train or util.get_cuda_memory_allocated() > self.memory_limit):
          if train:
            total_loss.backward(retain_graph=True)
          cpu_loss += total_loss.item()
          total_loss = None
    if total_loss is not None:
      cpu_loss += total_loss.item()
      if train:
        total_loss.backward(retain_graph=True)
    return cpu_loss

  def compute_attach_stats(self, best_cluster_idx, gold_cluster_id):
    if best_cluster_idx != 0:
      if (best_cluster_idx == gold_cluster_id):
        self.correct_attach += 1
      else:
        self.wrong_attach += 1
    else:
      if gold_cluster_id == 0:
        self.correct_new += 1
      else:
        self.wrong_new += 1

  def get_merge_fn(self, method, dropout):
    # move to config
    span_emb_size = self.span_scorer.spanFFNN.layers[0].weight.shape[1]
    if method == "mlp":
      self.mlp = util.FFNN(2 * span_emb_size, 3000, 1, dropout, output_dim=span_emb_size)
      merge_fn = lambda c, s, _: (self.mlp(torch.cat([c.emb, s.emb])), None)
    elif method == "alpha":
      self.mlp = util.FFNN(2 * span_emb_size, 300, 1, dropout, output_dim=1)
      def sigmoid_interpolate(c, s, score):
        alpha = torch.sigmoid(self.mlp(torch.cat([c.emb, s.emb])))
        return (alpha * c.emb + ((1 - alpha) * s.emb), alpha)
      merge_fn = sigmoid_interpolate
    else:
      merge_fn = MERGE_NAMES[method]
    return merge_fn


  def forward(self, segment, model_data, clusters, start_idx, mask, train=False):
    doc_embs = self.encoder(segment)
    if mask is not None:
      doc_embs = torch.index_select(doc_embs, 1, torch.tensor(mask).to(self.device))
    if not self.finetune or not train:  # not detaching is memory expenisve
      doc_embs = doc_embs.detach()

    if self.use_gold_spans:
      if "gold_spans" in model_data:
        gold_spans = model_data["gold_spans"]
      else:
        # Assume gold spans are just the cluster spans
        gold_spans = util.flatten(model_data["clusters"])
    else:
      gold_spans = None
    top_spans = self.get_top_spans(model_data["sentence_map"], doc_embs,
                                   start_idx, given_spans=gold_spans)
    if len(top_spans) == 0:
      # Nothing to update
      return
    genre_emb = self.genre_embedder(model_data["doc_key"][:2])
    num_words = doc_embs.shape[1]
    if mask is None:
      sentences = segment
    else:
      sentences = model_data["sentences"][start_idx: start_idx+num_words]
    segment_map = model_data["sentence_map"][start_idx:start_idx+num_words]
    sent_gen = util.get_sentence_iter(sentences, segment_map, top_spans, start_idx, genre_emb)
    return self.resolve_local(clusters, sent_gen, model_data["antecedent_map"], train=train)

  def clear_cache(self, clusters, idx):
    clusters.clear_cache(idx, self.evict_fn)
