import torch
from torch_scores import create_from_scratch
import util
from cluster import Cluster, ClusterList
from merge_functions import MERGE_NAMES
# from pytorch_memlab import profile
import numpy as np

class Incremental(torch.nn.Module):
  def __init__(self, config):
    super(Incremental, self).__init__()
    (self.encoder, self.genre_embedder,
     self.span_scorer, self.pair_scorer,
     self.device) = create_from_scratch(config)
    self.set_threshold(config["threshold"])
    self.max_span_width = config["max_span_width"]
    self.memory_limit = config["memory_limit"]
    self.negative_sample_rate = config["negative_sample_rate"]
    self.use_gold_spans = config["use_gold_spans"]
    self.teacher_forcing_rate = config["teacher_forcing_rate"]
    self.merge_fn = self.get_merge_fn(config["method"], config["dropout"])
    self.debug_embs = config["debug_embs"]
    self.cluster = lambda x: Cluster(x, self.merge_fn, self.debug_embs)
    self.finetune = config["finetune"]
    self.mention_classifier = config["mentions"]
    self.singleton = config["singleton_eval"]
    # debug
    self.correct_new = 0
    self.correct_attach = 0
    self.wrong_new = 0
    self.wrong_attach = 0
    self.loss_count = 0
    self.sampled_loss_count = 0
    # important for moving threshold to gpu
    self.to(self.device)
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

  def update_clusterlist(self, clusterlist, cluster, score, best_cluster_idx, gold_cluster_id, train):
    # Retrieve cluster if already classified
    best_cluster_idx = clusterlist.span_to_cluster.get((cluster.start, cluster.end),
                                                       best_cluster_idx)
    # Update the ClusterList structure
    if not train or (np.random.uniform() > self.teacher_forcing_rate) or best_cluster_idx == 0:
      if self.singleton and cluster.score < self.threshold and self.mention_classifier:
        pass
      elif best_cluster_idx == 0:
        clusterlist.append(cluster)
      else:
        clusterlist.merge(int(best_cluster_idx), cluster, score=score)
    else: # Use teacher forcing
      if gold_cluster_id == 0:
        clusterlist.append(cluster)
      else:
        clusterlist.merge(gold_cluster_id, cluster, score=score)

  def resolve_local(self, clusters, sent_gen, gold_map, train=False, metrics=True):
    total_loss = None
    cpu_loss = 0
    num_spans = 0
    for sent_idx, genre, cluster_list in sent_gen:
      # print (f"{sent_idx}: {len(cluster_list)} spans (so far {num_spans})")
      for cluster in cluster_list:
        gold_span = gold_map.get((cluster.first.start, cluster.first.end), None)
        gold_cluster_ids = clusters.get_cluster_ids(gold_span, original=(cluster.first.start, cluster.first.end)) # in range(0, num_clusters + 1)
        gold_cluster_id = max(gold_cluster_ids) # pick most recent gold cluster

        # Compute mention-level loss here (it runs over clusters now), short circuit if possible
        if self.mention_classifier:
          p_mention = torch.log_softmax(torch.cat([self.threshold, cluster.score.view(1)]), dim=0)
          gold_span_label = 0 if gold_span is None else 1
          span_loss = -1 * p_mention[gold_span_label]

          # Shortcircuit: If we know/predict it has no antecedent at training/inference, stop now.
          do_cluster_scoring = ((train and gold_span_label == 1) or
                                (not train and cluster.score > self.threshold))
        else:
          span_loss = None
          do_cluster_scoring = True
          # Might want to skip if there are too many negative spans
          if gold_span is not None:
            self.loss_count += 1
          elif np.random.uniform() < self.negative_sample_rate:
            self.sampled_loss_count += 1
          else:
            do_cluster_scoring = False # Just skip everything

        # Compute cluster/linking loss. All spans making it to this step will be written to predictions file.
        if do_cluster_scoring:
          num_spans += cluster.size
          # Edge case for first mention
          if len(clusters) == 0:
            self.update_clusterlist(clusters, cluster, cluster.score, 0, 0, train)
            cluster_loss = None
          else:
            cluster_embs = [c.emb for c in clusters]
            antecedent_starts = torch.tensor([(c.start/2.0 + c.end/2.0) for c in clusters], device=self.device)
            computed_offsets = (cluster.start + cluster.end)/2.0 - antecedent_starts  # type: int - tensor -> tensor
            pairwise_scores = self.pair_scorer(cluster.emb, cluster_embs, computed_offsets, genre)  # [1,1] score

            # With a mention classifier, we don't add the cluster.score (which is already used).
            # Without it, we can interpret cluster.score as an additional feature of the pairwise scorer.
            if self.mention_classifier:
              scores = pairwise_scores[0]
            else:
              scores = pairwise_scores[0] + cluster.score

            cluster_scores = torch.cat([self.threshold, scores])
            p_cluster = torch.log_softmax(cluster_scores, dim=0) # [num_clusters + 1]
            p_score, best_cluster_idx = torch.min(-1 * p_cluster, dim=0)
            cluster_loss = -1 * torch.logsumexp(p_cluster[gold_cluster_ids], dim=0) # log gold probabilities

            self.update_clusterlist(clusters, cluster, p_score, best_cluster_idx, gold_cluster_id, train)

            # Stats on updates
            if (not train) and metrics:
              self.compute_attach_stats(best_cluster_idx, gold_cluster_id)
        else:
          cluster_loss = None

        # Update total_loss to a single final loss
        total_loss = util.safe_add(total_loss,
                                   util.safe_add(span_loss, cluster_loss))

        # Check if we need to run backward() for memory reasons, only a concern in training
        # Get_memory is a bit expensive so we only check that part during training
        if not train or util.get_cuda_memory_allocated() > self.memory_limit:
          if train and total_loss is not None:
            total_loss.backward(retain_graph=True)
          cpu_loss = float(util.safe_add(cpu_loss, total_loss)) # float() needed to detach total_loss
          total_loss = None
    # Flush at end of program
    if train and total_loss is not None:
      total_loss.backward(retain_graph=True)
    cpu_loss = float(util.safe_add(cpu_loss, total_loss)) # float() needed to detach total_loss
    return cpu_loss

  def compute_attach_stats(self, best_cluster_idx, gold_cluster_id):
    if best_cluster_idx != 0:
      if best_cluster_idx == gold_cluster_id:
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
        # alpha = self.mlp(torch.cat([c.emb, s.emb]))
        # print (f"{alpha.item():.2f} * {c.bracket_string()} + {1-alpha.item():.2f} * {s.bracket_string()}")
        return (alpha * c.emb + ((1 - alpha) * s.emb), alpha)
      merge_fn = sigmoid_interpolate
    else:
      merge_fn = MERGE_NAMES[method]
    return merge_fn

  def forward(self, segment, model_data, clusters, start_idx, mask, train=False, metrics=True, consolidate=True):
    """
    forward() function called in trainer and inference. Performs a masking and
    sets up data and span iterator for the resolve_local() function

    Args:
      segment: list of tokens for this segment
      model_data: dict corresponding to the segment being evaluated
      clusters: current predicted clusters (should begin as an empty list)
      start_idx: start index of this segment (with respect to the document)
      mask: Typically None. If it is provided, matches segment and tells which tokens to ignor (e.g. to simulate the "overlap" version of coref model).
      train: whether the current call is in training (passed to resolve_local)
      metrics: whether to compute and print some metrics (passed to resolve_local)
      consolidate: whether to consolidate the provided clusters (if it was not empty)
    """
    # Usually, this "consolidate" step is a no-op since the given clusters are empty.
    if consolidate:
      genre_emb = self.genre_embedder(model_data["doc_key"][:2])
      # Consolidate given clusters
      new_clusters = ClusterList()
      cons_loss = self.resolve_local(new_clusters, [(None, genre_emb, clusters)], model_data["antecedent_map"], train=train, metrics=False)
      clusters.update(new_clusters)
    else:
      cons_loss = 0.0
    doc_embs = self.encoder(segment, f"{model_data['doc_key']}_{start_idx}")
    if mask is not None:
      doc_embs = torch.index_select(doc_embs, 1, torch.tensor(mask).to(self.device))
    if not self.finetune or not train:  # not detaching is memory expenisve
      doc_embs = doc_embs.detach()

    if self.use_gold_spans:
      if "gold_spans" in model_data:
        gold_spans = model_data["gold_spans"]
      else:
        # Assume gold spans are just the cluster spans
        # also dedup
        gold_spans = list({tuple(x) for x in util.flatten(model_data["clusters"])})
    else:
      gold_spans = None
    top_spans = self.get_top_spans(model_data["sentence_map"], doc_embs,
                                   start_idx, given_spans=gold_spans)
    if len(top_spans) == 0:
      # Nothing to update
      return 0.0
    num_words = doc_embs.shape[1]
    if mask is None:
      sentences = segment
    else:
      sentences = model_data["sentences"][start_idx: start_idx+num_words]
    segment_map = model_data["sentence_map"][start_idx:start_idx+num_words]
    genre_emb = self.genre_embedder(model_data["doc_key"][:2])
    sent_gen = util.get_sentence_iter(sentences, segment_map, top_spans, start_idx, genre_emb, self.cluster)
    spans_loss = self.resolve_local(clusters, sent_gen, model_data["antecedent_map"], train=train, metrics=metrics)
    return cons_loss + spans_loss

  def clear_cache(self, clusters, idx):
    clusters.clear_cache(idx, self.evict_fn)

  def set_threshold(self, threshold):
    self.threshold = torch.nn.Parameter(data=torch.Tensor([threshold]).to(self.device),
                                        requires_grad=False)
