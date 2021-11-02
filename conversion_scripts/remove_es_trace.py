import sys
import json

old_file = open(sys.argv[1], 'r')
new_file = sys.argv[2]

def fix_clusters(cluster, traces, rev_map):
    new_spans = []
    for span in cluster:
        if span[0] == span[1] and span[0] in traces:
            continue
        elif span[0] in traces:
            new_spans.append([rev_map[span[0] + 1], rev_map[span[1]]])
        elif span[1] in traces:
            new_spans.append([rev_map[span[0]], rev_map[span[1] - 1]])
        else:
            new_spans.append([rev_map[span[0]], rev_map[span[1]]])
    return new_spans

def remove_trace(jsondict):
    tokens = [tok for sent in jsondict["sentences"] for tok in sent]
    traces = []
    new_idx = []
    for i, tok in enumerate(tokens):
        if tok == "\u2581_":
            traces.append(i)
        else:
            new_idx.append(i)
    rev_map = {j: i for i, j in enumerate(new_idx)}
    # Fix clusters
    new_spans = [fix_clusters(c, traces, rev_map) for c in jsondict["clusters"]]
    new_clusters = [c for c in new_spans if len(c) > 0]
    # Fix sentence
    sentences = [[word for word in sentence if word != "\u2581_"] for sentence in jsondict["sentences"]]
    # Fix sentence_map - this should be correct
    sentence_map = [u for i, u in enumerate(jsondict["sentence_map"]) if i not in traces]
    # Fix subtoken map
    subtokens = []
    offset = 0
    for i, subtok_idx in enumerate(jsondict["subtoken_map"]):
        if i in traces:
            offset += 1
        else:
            subtokens.append(subtok_idx - offset)
    return {
        "doc_key": jsondict["doc_key"],
        "sentences": sentences,
        "clusters": new_clusters,
        "sentence_map": sentence_map,
        "subtoken_map": subtokens,
        "langauge": "spanish",
    }


docs = [json.loads(l) for l in old_file]
output = open(new_file, "w+")
for line in docs:
    output.write(json.dumps(remove_trace(line)) + "\n")
