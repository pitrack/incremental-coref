

class SpanLike(object):
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

  def detach_(self):
    self.emb = self.emb.detach_()
    self.score.detach_()

  def as_string(self):
    return self.sentence[self.start:self.end + 1]

  def __str__(self):
    return str(self.meta)


class Cluster(SpanLike):
  def __init__(self, span, merge_fn):
    super(Cluster, self).__init__(span.start, span.end, span.sentence, span.sentence_offset)
    self.size = 1.0
    self.scores = [span.score]
    self.spans = [span.meta]
    self.merge_fn = merge_fn # ["mean", "first", "last", "exp", "mlp"]
    # Initial cluster
    self.emb = span.emb.clone()
    self.span_embs = [] # empty; [self.emb] reserved for debugging and analysis
    self.merge_data = [1.0]

  def merge(self, span, score=0.0):
    self.emb, metadata = self.merge_fn(self, span, score)
    # only append for debug visualization.
    # self.span_embs.append(span.emb)
    self.renormalize(metadata)

    self.start = span.start
    self.end = span.end
    self.sentence = span.sentence
    self.sentence_offset = span.sentence_offset
    self.scores.append(span.score)
    self.spans.append(span.meta)
    self.size += 1

  def renormalize(self, metadata):
    alpha = metadata.item()
    new = 1.0 - alpha
    self.merge_data = [weight * alpha for weight in self.merge_data]
    self.merge_data.append(new)

  def detach_(self):
    self.emb.detach_()
    self.scores = [score.detach() for score in self.scores]
    [emb.detach() for emb in self.span_embs]

  def cpu_(self):
    self.emb = self.emb.cpu()
    self.scores = [score.to("cpu") for score in self.scores]

  def as_list(self):
    return [[int(span.start), int(span.end)] for span in self.spans]

  def __len__(self):
    return len(self.spans)

  def __str__(self):
    return (",".join([str(span) for span in self.spans]))

"""The graph"""
class ClusterList(list):
  def __init__(self):
    # 0th cluster is the empty antecedent
    self.clusters = [] # active
    self.cpu_clusters = [] # cpu
    self.span_to_cluster = {}
    self.num_clusters = len(self.clusters)

  def append(self, cluster):
    self.clusters.append(cluster)
    self.span_to_cluster[(cluster.start, cluster.end)] = self.num_clusters + 1
    self.num_clusters += 1

  def clear_cache(self, idx, evict_fn):
    # evict_fn has typecluster, idx -> Bool
    # detach singletons, move to cpu
    new_cluster_list = []
    num_clusters = 0
    span_to_cluster = {}
    for cluster in self.clusters:
      if evict_fn(cluster, idx):
        cluster.cpu_()
        self.cpu_clusters.append(cluster)
      else:
        new_cluster_list.append(cluster)
        span_to_cluster[(cluster.start, cluster.end)] = num_clusters + 1
        num_clusters += 1
    self.clusters = new_cluster_list
    self.span_to_cluster = span_to_cluster
    self.num_clusters = len(self.clusters)

  def detach_(self):
    for cluster in self.clusters:
      cluster.detach_()

  def cpu_(self):
    for cluster in self.clusters:
      cluster.cpu_()

  def merge_span(self, best_idx, span, score=0.0):
    # These indices are shifted up by one
    self.clusters[best_idx - 1].merge(span, score=score)
    self.span_to_cluster[(span.start, span.end)] = best_idx

  def get_cluster_id(self, span):
    return self.span_to_cluster.get(span, 0)

  def check_invariants(self):
    assert(len(self.clusters) == self.num_clusters)
    assert(max(self.span_to_cluster.values(), default=0) == self.num_clusters)

  def finalize_clusters(self):
    self.clusters = sorted(self.clusters + self.cpu_clusters, key=lambda c: -c.size)

  def get_clusters(self, condensed=False, print_clusters=True):
    "Mostly for printing purposes"
    def maybe_print(s):
      if print_clusters:
        print(s)
    self.finalize_clusters()
    extracted_clusters = []
    for cluster in self.clusters:
      current_cluster = []
      maybe_print("=" * 50)
      for span in cluster.spans:
        if not condensed:
          maybe_print(span.bracket_string())
        else:
          maybe_print("{}, {}: {}".format(span.start, span.end, span.bracket_string()))
          maybe_print(f"{span.__dict__}")
        current_cluster.append((span.start, span.end))
      extracted_clusters.append(tuple(current_cluster))
    return extracted_clusters

  def get_cluster_embs(self):
    return [(i, c.span_embs) for i, c in enumerate(self.clusters)]

  def as_list(self):
    return [cluster.as_list() for cluster in self.clusters]

  def __iter__(self):
    return iter(self.clusters)

  def __len__(self):
    return len(self.clusters)

  def __repr__(self):
    return str(self.clusters)
