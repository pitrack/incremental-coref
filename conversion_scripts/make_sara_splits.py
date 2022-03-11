import sys
import json

f = open(sys.argv[1], 'r')
train, dev, test = [], [], []


docs = [json.loads(l) for l in f]
for i in range(7):
  train.append("\n".join([json.dumps(doc) for doc in docs if doc["split"] not in [i, (i+1) % 7]]))
  dev.append("\n".join([json.dumps(doc) for doc in docs if doc["split"] == i]))
  test.append("\n".join([json.dumps(doc) for doc in docs if doc["split"] == (i+1) % 7]))

unattested = [doc for doc in docs if doc["split"] not in [0, 1, 2, 3, 4, 5, 6, 7]]
print (unattested)
import pdb; pdb.set_trace()

everything = list(zip(train, dev, test))

for i, (train_split, dev_split, test_split) in enumerate(everything):
  train_f =open(f"train.{i}.jsonlines", 'w+')
  dev_f =open(f"dev.{i}.jsonlines", 'w+')
  test_f =open(f"test.{i}.jsonlines", 'w+')

  train_f.write("".join(train_split))
  dev_f.write("".join(dev_split))
  test_f.write("".join(test_split))
