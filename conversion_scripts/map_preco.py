import json
import sys

input_file = open(sys.argv[1], 'r')
output_file = open(sys.argv[2], 'w+')
for line in input_file:
    d = json.loads(line)
    clusters = d["mention_clusters"]
    sentences = d["sentences"]
    len_map = [len(sentence) for sentence in sentences]
    cum_sum = [sum(len_map[:i]) for i in range(len(sentences))]
    remap_clusters = lambda x: [cum_sum[x[0]] + x[1],
                                cum_sum[x[0]] + x[2]]
    mapped_clusters = [[remap_clusters(span) for span in cluster]
                       for cluster in clusters]
    d["clusters"] = mapped_clusters
    # print (mapped_clusters)
    output_file.write(json.dumps(d) + "\n")
