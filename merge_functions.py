"""
Contains several implementations of possible merge functions

All functions have type

Cluster * Span * score --> emb

"""


def first(cluster, span, score):
  return _alpha_weighted(1.0, cluster.emb, span.emb)

def last(cluster, span, score):
  return _alpha_weighted(0.0, cluster.emb, span.emb)

def mean(cluster, span, score):
  alpha = (cluster.size) / (1 + cluster.size)
  return _alpha_weighted(alpha, cluster.emb, span.emb)

def exp(cluster, span, score):
  return _alpha_weighted(0.5, cluster.emb, span.emb)

def _alpha_weighted(alpha, emb1, emb2):
  return (alpha * emb1 + (1.0 - alpha) * emb2, alpha)

MERGE_NAMES = {
    "mean": mean,
    "first": first,
    "last": last,
    "exp": exp,
}
