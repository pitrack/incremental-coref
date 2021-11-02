"""
Clusters and Span data structure file
"""

class SpanLike():
  def __init__(self, start, end, sentence, sentence_offset):
    self.start = start # true indices
    self.end = end
    self.sentence = sentence
    self.sentence_offset = sentence_offset # adjusted for sentence

  def bracket_print(self):
    total = []
    for i, tok in enumerate(self.sentence):
      if i + self.sentence_offset == self.start:
        total.append("[")
      total.append(tok)
      if i + self.sentence_offset == self.end:
        total.append("]")
    print(" ".join(total))

  def bracket_string(self):
    return self.sentence[self.start - self.sentence_offset:
                         self.end - self.sentence_offset + 1]

  def detach_(self):
    pass

  def __str__(self):
    return "[{}, {}]".format(self.start, self.end)

  def __repr__(self):
    return str(self)

  def __eq__(self, other):
    return self.start == other.start and self.end == other.end

  def __lt__(self, other):
    return ((self.start < other.start) or
            (self.start == other.start and self.end < other.end))


class Span(SpanLike):
  def __init__(self, emb, start, end, offset, sentence, score):
    super(Span, self).__init__(start, end, sentence, offset)
    self.emb = emb
    self.score = score
    self.meta = SpanLike(start, end, sentence, offset)
    self.original_device = emb.device

  def detach_(self):
    self.emb = self.emb.detach_()
    self.score.detach_()

  def as_string(self):
    return self.sentence[self.start:self.end + 1]

  def __str__(self):
    return str(self.meta)


class Cluster(SpanLike):
  def __init__(self, span, merge_fn, debug_embs):
    super(Cluster, self).__init__(span.start, span.end, span.sentence, span.sentence_offset)
    self.first = SpanLike(span.start, span.end, span.sentence, span.sentence_offset)
    self.size = 1.0
    self.scores = [span.score]
    self.score = span.score
    self.spans = [span.meta]
    self.merge_fn = merge_fn # ["mean", "first", "last", "exp", "mlp"]
    # Initial cluster
    self.emb = span.emb.clone()
    self.debug_embs = debug_embs
    if not self.debug_embs:
      self.span_embs = [] # [self.emb] reserved for debugging and analysis
    else:
      self.span_embs = [self.emb.tolist()]
    self.merge_data = [1.0]
    self.probs = [1.0]
    self.original_device = span.original_device

  def merge(self, cluster, score=0.0):
    self.emb, metadata = self.merge_fn(self, cluster, score)
    # for debug visualization only
    if self.debug_embs:
      self.span_embs.append(cluster.emb.tolist())
    self.renormalize(metadata)
    self.start = cluster.start
    self.end = cluster.end
    self.sentence = cluster.sentence
    self.sentence_offset = cluster.sentence_offset
    # first is always the earliest
    if (cluster.first.start < self.first.start or
        (cluster.first.start == self.first.start and cluster.first.end < self.first.end)):
      self.first = SpanLike(cluster.first.start, cluster.first.end,
                            cluster.first.sentence, cluster.first.sentence_offset)
    self.scores.extend(cluster.scores)
    self.spans.extend(cluster.spans)
    self.score = (self.size * self.score + cluster.size * cluster.score) / (self.size + cluster.size)
    self.size += cluster.size


  def renormalize(self, metadata):
    alpha = float(metadata)
    new = 1.0 - alpha
    self.merge_data = [weight * alpha for weight in self.merge_data]
    self.probs = self.merge_data
    self.merge_data.append(new)

  def detach_(self):
    self.emb.detach_()
    self.scores = [score.detach() for score in self.scores]
    self.score = self.score.detach()
    [emb.detach() for emb in self.span_embs]

  def cpu_(self):
    self.emb = self.emb.cpu()
    self.scores = [score.to("cpu") for score in self.scores]

  def reset(self):
    """ Reset cluster to a single prior """
    self.size = 0.0
    self.spans = [] # Pick first one as canonical mention? Or maybe it should be max
    self.scores = []
    self.merge_data = [1.0]
    self.probs = [1.0]
    self.emb = self.emb.to(self.original_device)
    # Set to 0 when reset, to simulate background context
    # Or use first since that's the "official" name of the cluster?
    self.start = 0 #self.first.start
    self.end = 0 # self.first.end

  def as_list(self):
    return [[int(span.start), int(span.end)] for span in self.spans]

  def __iter__(self):
    return iter(self.spans)

  def __len__(self):
    return len(self.spans)

  def __str__(self):
    return (",".join([str(span) for span in self.spans]))

class ClusterList(list):
  """
  The graph that is built incrementally
  """
  def __init__(self):
    # 0th cluster is the empty antecedent
    self.clusters = [] # active
    self.cpu_clusters = [] # cpu
    self.span_to_cluster = {}
    self.num_clusters = len(self.clusters)

  def update(self, clusterlist):
    self.clusters = clusterlist.clusters
    self.cpu_clusters = clusterlist.cpu_clusters
    self.span_to_cluster = clusterlist.span_to_cluster
    self.num_clusters = clusterlist.num_clusters
    for cluster in self.clusters:
      cluster.reset()

  def append(self, cluster):
    self.clusters.append(cluster)
    for span in cluster:
      self.span_to_cluster[(span.start, span.end)] = self.num_clusters + 1
    self.num_clusters += 1

  def clear_cache(self, idx, evict_fn):
    # evict_fn has typecluster, idx --> Bool
    # detach singletons, move to cpu?
    """
    Move things to cpu_clusters if appropriate
    """
    new_cluster_list = []
    for cluster in self.clusters:
      if evict_fn(cluster, idx):
        cluster.cpu_()
        self.cpu_clusters.append(cluster)
      else:
        new_cluster_list.append(cluster)
    self.restrict_clusters(new_cluster_list)

  def detach_(self):
    for cluster in self.clusters:
      cluster.detach_()

  def cpu_(self):
    for cluster in self.clusters:
      cluster.cpu_()

  def merge(self, best_idx, cluster, score=0.0):
    # Note this is shifted up by one
    self.clusters[best_idx - 1].merge(cluster, score=score)
    for span in cluster:
      self.span_to_cluster[(span.start, span.end)] = best_idx

  def reset(self):
    """
    Keep only the best clusters, wipe everything else in the list.
    """
    # print([len(c) for c in self.clusters] + [len(c) for c in self.cpu_clusters])
    # Need to add cpu clusters to span_to_cluster
    for c in self.cpu_clusters:
      self.append(c)
      curr_idx = self.num_clusters
      for span in c.spans:
        self.span_to_cluster[(span.start, span.end)] = curr_idx
    clusters = []
    self.cpu_clusters = [] # We are doing a hard reset
    self.restrict_clusters(clusters)#, first_only=True)
    self.detach_()
    for cluster in self.clusters:
      cluster.emb = cluster.emb.to(cluster.original_device)
      # cluster.reset()

  def restrict_clusters(self, clusters, first_only=False):
    num_clusters = 0
    span_to_cluster = {}
    saved_cluster_idxs = {}
    for cluster in clusters:
      if first_only:
        span_to_cluster[(cluster.first.start, cluster.first.end)] = num_clusters + 1
      else:
        curr_span_idx = self.span_to_cluster[(cluster.first.start, cluster.first.end)]
        saved_cluster_idxs[curr_span_idx] = num_clusters + 1
      num_clusters += 1
    if not first_only:
      for span, c_idx in self.span_to_cluster.items():
        if c_idx in saved_cluster_idxs:
          span_to_cluster[span] = saved_cluster_idxs[c_idx]
    self.clusters = clusters
    self.num_clusters = num_clusters
    self.span_to_cluster = span_to_cluster
    # self.check_invariants()

  def get_cluster_id(self, span, original=0):
    if original in self.span_to_cluster:
      return self.span_to_cluster[original]
    return self.span_to_cluster.get(span,
                                    self.span_to_cluster.get(original, 0))

  def get_cluster_ids(self, spans, original=0):
    if spans is None:
      return [0]
    cluster_ids = []
    for span in spans:
      cluster_id = self.get_cluster_id(span, original=original)
      if cluster_id != 0:
        cluster_ids.append(cluster_id)
    if len(cluster_ids) == 0:
      return [0]
    return list(set(cluster_ids))

  def check_invariants(self):
    assert len(self.clusters) == self.num_clusters
    assert max(self.span_to_cluster.values(), default=0) == self.num_clusters
    for cluster in self.clusters:
      assert (cluster.first.start, cluster.first.end) in self.span_to_cluster
    # this doesn't need to be true if span_to_cluster contains ghosts
    # assert self.num_spans() == len(self.span_to_cluster)

  def finalize_clusters(self, filter_f=lambda x: x):
    sorted_total = sorted(self.clusters + self.cpu_clusters, key=lambda c: -c.size)
    return [cluster for cluster in sorted_total if filter_f(cluster)]

  def get_clusters(self, singleton_eval, condensed=False, print_clusters=True, ):
    "Mostly for printing purposes"
    def maybe_print(s):
      if print_clusters:
        print(s)
    sorted_total = self.finalize_clusters(lambda x: len(x) > (0 if singleton_eval else 1))
    response_dict = {}
    extracted_clusters = []
    cluster_embs = []
    span_embs = []
    for cluster in sorted_total:
      current_cluster = []
      maybe_print("=" * 50)
      for span in cluster.spans:
        if not condensed:
          maybe_print(span.bracket_string())
        else:
          maybe_print("{}, {}: {}".format(span.start, span.end, span.bracket_string()))
          maybe_print(f"{span.__dict__}")
        current_cluster.append((span.start, span.end))
      span_embs.append(cluster.span_embs)
      cluster_embs.append(cluster.emb.tolist())
      extracted_clusters.append(current_cluster)
    response_dict["clusters"] = extracted_clusters
    response_dict["span_embs"] = span_embs
    response_dict["cluster_embs"] = cluster_embs
    return response_dict

  def get_cluster_embs(self):
    return [(i, c.span_embs) for i, c in enumerate(self.clusters)]

  def ugly_print(self, limit=1000):
    clusters = sorted(self.clusters, key=lambda x: x.size)
    print("\n\n".join(
      ["\t\t".join(
        [" ".join(s.bracket_string()) for s in c.spans])
       for c in clusters[:limit]]
    ))

  def as_list(self, singleton_eval):
    return [cluster.as_list() for cluster in
            self.finalize_clusters(lambda x: len(x) > (0 if singleton_eval else 1))]

  def num_spans(self, total=False):
    if total:
      cluster_iter = iter(self.clusters + self.cpu_clusters)
    else:
      cluster_iter = iter(self.clusters)
    return sum([len(cluster) for cluster in cluster_iter])

  def __iter__(self):
    return iter(self.clusters)

  def __len__(self):
    return len(self.clusters)

  def __repr__(self):
    return str(self.clusters)
