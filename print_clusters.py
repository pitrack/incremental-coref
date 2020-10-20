import json
import sys
import util
import argparse


def print_clusters(args):
    f = open(args.preds)
    for i, line in enumerate(f):
      data = json.loads(line)
      text = util.flatten(data['sentences'])
      for ci, cluster in enumerate(data[args.key]):
        spans = [text[s:e+1] for s,e in cluster]
        if len(spans) > args.size:
            print(i, ci, len(spans), spans)
      if i > 5:
        break

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-p", "--preds", required=True)
    parser.add_argument("-k", "--key", default="predicted_clusters")
    parser.add_argument("-s", "--size", default=0, type=int)
    args = parser.parse_args()
    print_clusters(args)
