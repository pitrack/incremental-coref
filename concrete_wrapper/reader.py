import sys
from collections import namedtuple, defaultdict
import concrete
from concrete.util import CommunicationReader
import numpy as np
import json
Mention = namedtuple("Mention", "text start end sentence entityType confidence uuid")

def get_entities(comm, entity_tool):
  """
  Returns:
  list of concrete.Entity objects
  """
  entity_set_index = concrete.util.metadata.get_index_of_tool(
    comm.entitySetList, entity_tool)
  if entity_set_index == -1:
    print(f"Could not find EntitySet with tool name {entity_tool}")
    return []
  else:
    return comm.entitySetList[entity_set_index].entityList

def comm_to_dict(comm, entity_tool):
  output_dict = {}
  output_dict['doc_id'] = comm.id

  # Assume single section in sectionList
  sentences = []
  sentence_tokenization_uuids = {}
  offset_dicts = {
    "idx_to_char_offsets": {},
    "char_offsets_to_idx": {}
  }
  # Read comm into list of tokenized sentences.
  for i, sentence in enumerate(comm.sectionList[0].sentenceList):
    sentence_tokenization_uuids[i] = sentence.tokenization.uuid.uuidString
    sentence_text = []
    idx_to_char_offsets = {}
    char_offsets_to_idx = {}
    for token in sentence.tokenization.tokenList.tokenList:
      token_idx = token.tokenIndex
      token_start = token.textSpan.start
      token_ending = token.textSpan.ending
      token = token.text
      sentence_text.append(token)
      idx_to_char_offsets[token_idx] = (token_start, token_ending)
      char_offsets_to_idx[(token_start, token_ending)] = token_idx
    offset_dicts["idx_to_char_offsets"][i] = idx_to_char_offsets
    offset_dicts["char_offsets_to_idx"][i] = char_offsets_to_idx
    sentences.append(sentence_text)

  output_dict["sentences"] = sentences
  output_dict["offset_dicts"] = offset_dicts

  # Compute offsets
  sentence_offsets = np.cumsum([0] + [len(s) for s in sentences])
  tokenization_to_sent = {uuid:idx
                          for idx, uuid in sentence_tokenization_uuids.items()}

  output_dict['entity_set_list'] = []
  # Read through entity mention set list
  mention_list = []
  mention_uuid_map = {}
  mention_skip_map = {}
  for ms_idx, mention_set in enumerate(comm.entityMentionSetList):
    # print ("{} mention_list: {}".format(ms_idx, len(mention_set.mentionList)))
    for mention in mention_set.mentionList:
      tokens = mention.tokens.tokenIndexList
      tokenizationId = mention.tokens.tokenizationId
      sentId = tokenization_to_sent[mention.tokens.tokenizationId.uuidString]
      sent_toks = [sentences[sentId][idx] for idx in tokens]
      m = Mention(text=mention.text,
                  start=min(tokens),
                  end=max(tokens),
                  sentence=sentId,
                  entityType=mention.entityType,
                  confidence=mention.confidence,
                  uuid=mention.uuid)
      mention_list.append(m)
      mention_uuid_map[mention.uuid.uuidString] = m
  output_dict['mentions'] = [(int(sentence_offsets[m.sentence] + m.start),
                              int(sentence_offsets[m.sentence] + m.end))
                              for m in mention_list]

  # Convert Mention to doc-level (start, end) and update mapping
  mention_map = defaultdict(list)
  for m in mention_list:
    start = int(sentence_offsets[m.sentence] + m.start)
    end = int(sentence_offsets[m.sentence] + m.end)
    mention_map[(start, end)].append(m)
  output_dict["mention_map"] = mention_map


  output_dict["clusters"] = []
  # Get entity set list using entity_tool
  if entity_tool is not None:
    entity_list = get_entities(comm, entity_tool)
    uuid_clusters = []
    print (f"Found entity list with {len(entity_list)} entities")
    for entity in entity_list:
      if entity.mentionIdList:
        uuid_clusters.append(entity.mentionIdList)

    mention_count = 0
    clusters = []
    seen = set()
    for cluster in uuid_clusters:
      entity_list = []
      for mention_uuid in cluster:
        if mention_uuid.uuidString not in seen:
          seen.add(mention_uuid.uuidString)
        else:
          print(f"{mention_uuid} in two different clusters")
        m = mention_uuid_map[mention_uuid.uuidString]
        start = int(sentence_offsets[m.sentence] + m.start)
        end = int(sentence_offsets[m.sentence] + m.end)
        entity_list.append([start, end])
      if entity_list:
        clusters.append(entity_list)
    # Ensure every mention is used in exactly one cluster
    assert(len(mention_uuid_map) == len(seen))
    output_dict["clusters"] = clusters

  return (output_dict, comm)

def make_data_iter(path, entity_tool):
  for (comm, filename) in CommunicationReader(path):
    print (f"Entity_tool: {entity_tool}")
    yield comm_to_dict(comm, entity_tool)

if __name__ == "__main__":
  input_comms = sys.argv[1]
  output_file = sys.argv[2]
  if len(sys.argv) > 3:
    entity_tool = sys.argv[3]
  else:
    entity_tool = None
  examples_iter = make_data_iter(input_comms, entity_tool)
  output_file = open(output_file, 'w+')
  for example, _ in examples_iter:
    clean_version = {
      "sentences": example["sentences"],
      "doc_key": example["doc_id"],
    }
    if example["clusters"]:
      clean_version["clusters"] = example["clusters"]
    else:
      clean_version["clusters"] = [[span] for span in set(example["mentions"])]

    num_clusters = len(clean_version["clusters"])
    num_mentions = sum([len(cluster) for cluster in clean_version["clusters"]])
    num_total_mentions = sum([len(mention) for mention in example["mention_map"].values()])
    print(f"Wrote {num_clusters} clusters and {num_mentions} mentions" +
          f" (out of {num_total_mentions}) to {clean_version['doc_key']}")
    output_file.write(json.dumps(clean_version) + "\n")
