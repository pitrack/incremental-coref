import re
import os
import sys
import json
import tempfile
import subprocess
import collections

from transformers import *

def flatten(l):
  return [item for sublist in l for item in sublist]

def right_endpoint(token_map, idx):
  if idx + 1 == len(token_map):
    print ("Error in retokenizing to json")
  else:
    return token_map[idx + 1] - 1

class DocumentState(object):
  def __init__(self, key):
    self.doc_key = key
    self.sentence_end = []
    self.token_end = []
    self.tokens = []
    self.subtokens = []
    self.info = []
    self.segments = []
    self.subtoken_map = []
    self.segment_subtoken_map = []
    self.sentence_map = []
    self.pronouns = []
    self.clusters = collections.defaultdict(list)
    self.coref_stacks = collections.defaultdict(list)
    self.speakers = []
    self.segment_info = []

  def finalize(self):
    sentence_map = get_sentence_map(self.segments, self.sentence_end)
    subtoken_map = flatten(self.segment_subtoken_map)
    reverse_subtoken_map = {}
    subtoken_map_iter = subtoken_map + [subtoken_map[-1] + 1] if len(subtoken_map) > 0 else subtoken_map
    for subtok, tok in enumerate(subtoken_map_iter):
      if tok not in reverse_subtoken_map:
        reverse_subtoken_map[tok] = subtok
    clusters = [[[reverse_subtoken_map[span[0]], right_endpoint(reverse_subtoken_map, span[1])] for span in cluster]
                  for cluster in self.clusters]
    # assert len(all_mentions) == len(set(all_mentions))
    num_words =  len(flatten(self.segments))
    # assert num_words == len(flatten(self.speakers))
    assert num_words == len(subtoken_map), (num_words, len(subtoken_map))
    assert num_words == len(sentence_map), (num_words, len(sentence_map))
    return {
      "doc_key": self.doc_key,
      "sentences": self.segments,
      "clusters": clusters,
      'sentence_map':sentence_map,
      "subtoken_map": subtoken_map,
    }


# def normalize_word(word, language):
#   if language == "arabic":
#     word = word[:word.find("#")]
#   if word == "/." or word == "/?":
#     return word[1:]
#   else:
#     return word

# first try to satisfy constraints1, and if not possible, constraints2.
# first try to satisfy constraints1, and if not possible, constraints2.
def split_into_segments(document_state, max_segment_len, constraints1, constraints2, sentences=False):
  current = 0
  previous_token = 0
  boundaries = [i for i, c1 in enumerate(constraints1) if c1]
  curr_sent = 0
  while current < len(document_state.subtokens):
    if not sentences:
      end = min(current + max_segment_len - 1 - 2, len(document_state.subtokens) - 1)
    else:
      curr_sent += max_segment_len
      end = min(current + 512 - 1 - 2, # max_segment_len is 512
                len(document_state.subtokens) - 1,
                boundaries[curr_sent] if curr_sent < len(boundaries) else 1000000)
    while end >= current and not constraints1[end]:
      end -= 1
    if end < current:
        end = min(current + max_segment_len - 1 - 2, len(document_state.subtokens) - 1)
        while end >= current and not constraints2[end]:
            end -= 1
        if end < current:
            raise Exception("Can't find valid segment")
    document_state.segments.append(['[CLS]'] + document_state.subtokens[current:end + 1] + ['[SEP]'])
    subtoken_map = document_state.subtoken_map[current : end + 1]
    document_state.segment_subtoken_map.append([previous_token] + subtoken_map + [subtoken_map[-1]])
    info = document_state.info[current : end + 1]
    document_state.segment_info.append([None] + info + [None])
    current = end + 1
    previous_token = subtoken_map[-1]

def get_sentence_map(segments, sentence_end):
  current = 0
  sent_map = []
  sent_end_idx = 0
  assert len(sentence_end) == sum([len(s) -2 for s in segments])
  for segment in segments:
    sent_map.append(current)
    for i in range(len(segment) - 2):
      sent_map.append(current)
      current += int(sentence_end[sent_end_idx])
      sent_end_idx += 1
    sent_map.append(current)
  return sent_map

def get_document(document_lines, tokenizer, language, segment_len, stats=None):
  document_state = DocumentState(document_lines[0])
  document_state.clusters = document_lines[2]
  word_idx = -1
  for line in document_lines[1]:
    row = list(line)
    sentence_end = (len(row) == 0)
    if not sentence_end:
      # assert len(row) >= 12
      word_idx += 1
      word = row[0] # normalize_word(row[0], language)
      subtokens = tokenizer.tokenize(word)
      document_state.tokens.append(word)
      document_state.token_end += ([False] * (len(subtokens) - 1)) + [True]
      if len(subtokens) == 0 or len(subtokens) > 100:
        print (f"Skipping token in sentence in {document_lines[0]}: {word, subtokens[:35]}...")
        # print (word_idx, line, document_lines)
        word = "-"
        subtokens = tokenizer.tokenize(word)
      for sidx, subtoken in enumerate(subtokens):
        document_state.subtokens.append(subtoken)
        info = None if sidx != 0 else (row + [len(subtokens)])
        document_state.info.append(info)
        document_state.sentence_end.append(False)
        document_state.subtoken_map.append(word_idx)
    else:
      document_state.sentence_end[-1] = True
  # split_into_segments(document_state, segment_len, document_state.token_end)
  # split_into_segments(document_state, segment_len, document_state.sentence_end)
  constraints1 = document_state.sentence_end if language != 'arabic' else document_state.token_end
  split_into_segments(document_state, segment_len, constraints1, document_state.token_end)
  if stats is not None:
    stats["max_sent_len_{}".format(language)] = max(max([len(s) for s in document_state.segments]), stats["max_sent_len_{}".format(language)])
  document = document_state.finalize()
  return document


def clean_spaces(sentences, clusters, tokenizer):
  """ Used for processing SARA. It's a bit specific"""
  offsets = {}
  curr_offset = 0
  new_sentences = []
  curr_token = 0
  # print(sentences, clusters)
  for sentence in sentences:
    new_sentence = []
    for word in sentence:
      subtokens = tokenizer.tokenize(word)
      if len(subtokens) == 0 or len(subtokens) > 100:
        # print(f"bad: {word}")
        curr_offset -= 1
        offsets[curr_token] = None
      else:
        new_sentence.append(word)
        offsets[curr_token] = curr_offset
      curr_token += 1
    new_sentences.append(new_sentence)
  # print(offsets)
  if clusters:
    clusters = [[[p[0] + offsets[p[0]], p[1] - 1 + offsets[p[1] - 1]]
                     for p in cluster]
                    for cluster in clusters]
  # print(new_sentences, clusters)
  return new_sentences, clusters

def minimize_partition(name, language, extension, labels, stats, tokenizer, seg_len, input_dir, output_dir, lines=True):
  input_path = "{}/{}.{}{}".format(input_dir, name, language, extension)
  output_path = "{}/{}.{}{}.jsonlines".format(output_dir, name, language, seg_len)
  count = 0
  documents = []
  with open(input_path, "r", encoding="utf-8") as input_file:
    if lines:
      file_lines = input_file.readlines()
    else:
      file_lines = [input_file.read()]
    for i, line in enumerate(file_lines):
      doc = json.loads(line)
      doc_key = doc["doc_key"] if "doc_key" in doc else doc["id"] if "id" in doc else str(i)
      text = []
      clusters = doc["clusters"] if "clusters" in doc else {}
      sentences = doc["sentences"] if "sentences" in doc else doc["full_text"]
      sentences, clusters = clean_spaces(sentences, clusters, tokenizer)
      for sentence in sentences:
        text.extend([[word] for word in sentence])
        text.append([])
      documents.append((doc_key, text, clusters, doc["split"]))
  with open(output_path, "w") as output_file:
    for document_lines in documents:
      document = get_document(document_lines, tokenizer, language, seg_len, stats=stats)
      document["split"] = document_lines[-1]
      output_file.write(json.dumps(document))
      output_file.write("\n")
      count += 1
  return count

def minimize_language(language, labels, stats, seg_len, input_dir, output_dir, model):
  if model == "bert":
    tokenizer = BertTokenizer.from_pretrained('bert-large-cased')
  elif model == "xlmr":
    tokenizer = XLMRobertaTokenizer.from_pretrained('xlm-roberta-large')
  total_count = 0
  files = os.listdir(input_dir)
  print (files)
  for file_name in files:
    print (file_name)
    if not file_name.endswith("jsonlines"):
      continue
    file_head = file_name.split(".jsonlines")[0]
    total_count += minimize_partition(file_head, language, "jsonlines", labels, stats, tokenizer, seg_len, input_dir, output_dir, lines=True)
  print("Wrote {} documents to {}".format(total_count, output_dir))


if __name__ == "__main__":
  input_dir = sys.argv[1]
  output_dir = sys.argv[2]
  model = sys.argv[3]
  labels = collections.defaultdict(set)
  stats = collections.defaultdict(int)
  if not os.path.isdir(output_dir):
    os.mkdir(output_dir)
  for seg_len in [512]:
    minimize_language("", labels, stats, seg_len, input_dir, output_dir, model)
  for k, v in labels.items():
    print("{} = [{}]".format(k, ", ".join("\"{}\"".format(label) for label in v)))
  for k, v in stats.items():
    print("{} = {}".format(k, v))
