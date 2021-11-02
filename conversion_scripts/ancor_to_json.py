import sys, os
from collections import defaultdict
import json

text_dir = sys.argv[1]
chains_dir = sys.argv[2]
mentions_dir = sys.argv[3]
output = sys.argv[4]
books = os.listdir(chains_dir)

output_file = open(output, 'w+')

def fix_bounds(token_dict, index, go_up=True):
    if index in token_dict:
        return token_dict[index]
    if go_up:
        return fix_bounds(token_dict, index + 1, go_up=go_up)
    else:
        return fix_bounds(token_dict, index - 1, go_up=go_up)

num_real_mentions = 0
num_mentions = 0
num_chains = 0
for book in books:
    try:
        tokens_file = open(text_dir + "/" + book, 'r')
    except:
        print (f"Skipping {book}")
        continue

    tokens_list = tokens_file.readlines()
    starts = {}
    ends = {}
    sent = []
    curr_doc_id = 0
    curr_doc = []
    doc_len = 0
    num_reals = 0
    for i, tokstr in enumerate(tokens_list):
        tokstr = tokstr.strip()
        sent_end = tokstr == ""
        if sent_end:
            curr_doc.append(sent)
            sent = []
        else:
            tok = tokstr.split("\t")
            sent.append(tok[3])
            starts[int(tok[1])] = doc_len
            ends[int(tok[1]) + int(tok[2])] = doc_len
            doc_len += 1

    if sent:
        curr_doc.append(sent)

    cluster_doc = open(chains_dir + "/" + book, 'r')
    clusters = defaultdict(list)
    seen_mentions = set()
    for line in cluster_doc:
        (mid, start, length, chain_id) = tuple([int(x) for x in line.strip().split()])
        left = fix_bounds(starts, start, go_up=True)
        right = fix_bounds(ends, start + length, go_up=False)
        if left > right:
            print (f"Died on {left}, {right}")
            right = left
        clusters[chain_id].append([left, right])
        if (left, right) not in seen_mentions:
            seen_mentions.add((left, right))
        else:
            print ("dupe")
            import pdb; pdb.set_trace()

    mentions_doc = open(mentions_dir + "/" + book, 'r')
    for i, line in enumerate(mentions_doc):
        (mid, start, length) = tuple([int(x) for x in line.strip().split()])
        num_real_mentions += 1
        num_reals += 1
        left = fix_bounds(starts, start, go_up=True)
        right = fix_bounds(ends, start + length, go_up=False)
        if left > right:
            print (f"And died on {left}, {right}")
            right = left
        if (left, right) not in seen_mentions:
            clusters[1000000 + i].append([left, right])


    num_chains += len(clusters)

    net_mentions = sum([len(c) for c in clusters.values()])
    num_mentions += net_mentions

    json_dict = {
        "doc_key": "ancor_" + book,
        "language": "russian",
        "sentences": curr_doc,
        "clusters": list(clusters.values()),
    }
    output_file.write(json.dumps(json_dict) + "\n")

print (num_real_mentions)
print (num_mentions)
print (num_chains)
