import sys
import json

f = open(sys.argv[1], 'r')
docs = list(f)

train, dev, test = [], [], []
for i in range(5):
  train.append(list(docs[:240]))
  dev.append(list(docs[240:320]))
  test.append(list(docs[320:400]))
  docs = docs[80:] + docs[:80]


docs = [json.loads(l) for l in f]
everything = list(zip(train, dev, test))

for i, (train_split, dev_split, test_split) in enumerate(everything):
  train_f =open(f"train.{i}.jsonlines", 'w+')
  dev_f =open(f"dev.{i}.jsonlines", 'w+')
  test_f =open(f"test.{i}.jsonlines", 'w+')

  train_f.write("".join(train_split))
  dev_f.write("".join(dev_split))
  test_f.write("".join(test_split))
