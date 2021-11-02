import sys
import json
from collections import defaultdict

groups_file = open(sys.argv[1], 'r')
tokens_file = open(sys.argv[2], 'r')
doc_file = open(sys.argv[3], 'r')
output_file = open(sys.argv[4], 'w+')
# tokens_list = csv.reader(tokens_file, delimiter="\t")

tokens_list = tokens_file.readlines()
groups_list = groups_file.readlines()
doc_list = doc_file.readlines()

documents = {}
starts = defaultdict(dict)
ends = defaultdict(dict)
doc_lens = defaultdict(int)
sent = []
curr_doc_id = 0
curr_doc = []

for i, tokstr in enumerate(tokens_list[1:]):
    tok = tokstr.strip().split("\t")
    doc_id = int(tok[0])
    if doc_id != curr_doc_id:
        # shift
        print (doc_id)
        if sent:
            curr_doc.append(sent)
        documents[curr_doc_id] = curr_doc
        curr_doc_id = doc_id
        curr_doc = []

    # if i < 31800 and i > 31400:
    #     print (i, doc_id, tok[1])
    sent_end = tok[5] == "SENT"
    sent.append(tok[3])
    starts[doc_id][int(tok[1])] = doc_lens[doc_id]
    ends[doc_id][int(tok[1]) + int(tok[2])] = doc_lens[doc_id]
    doc_lens[doc_id] += 1
    if sent_end:
        curr_doc.append(sent)
        sent = []

if sent:
    curr_doc.append(sent)
documents[curr_doc_id] = curr_doc
curr_doc_id = doc_id
curr_doc = []

clusters = defaultdict(list)
chains = defaultdict(set)
for groupstr in groups_list[1:]:
    group = groupstr.split("\t")
    doc_id = int(group[0])
    chain = group[3]
    start = int(group[5])
    end = int(group[5]) + int(group[6])
    if group[2] == "475746":
        end += 1 # there is an error there
    if int(chain) == 0:
        chain = doc_id * 100000 + 1
    chains[doc_id].add(int(chain))
    clusters[int(chain)].append([starts[doc_id][start],
                                 ends[doc_id][end]])

for docstr in doc_list[1:]:
    doc = docstr.split("\t")
    doc_id = int(doc[0])
    doc_key = "rucor_" + doc[1].replace("/", "_")
    language = "russian"
    tokens = documents[doc_id]
    doc_clusters = [clusters[cid] for cid in chains[doc_id]]
    json_dict = {
        "doc_key": doc_key,
        "language": language,
        "sentences": tokens,
        "clusters": doc_clusters,
    }
    output_file.write(json.dumps(json_dict) + "\n")
