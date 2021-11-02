import sys
from concrete.util import write_communication_to_file, now_timestamp
import concrete.entities.ttypes as cet
import concrete.metadata.ttypes as cmt
from concrete.util.concrete_uuid import AnalyticUUIDGeneratorFactory
from concrete.validate import validate_communication
import json
import concrete_wrapper.reader
from collections import Counter

def map_subtokens_to_tokens(token_nested_list, subtoken_map):
  if len(token_nested_list) == 0:
    return []
  if type(token_nested_list[0]) is list:
    return [map_subtokens_to_tokens(l, subtoken_map) for l in token_nested_list]
  else:
    return [subtoken_map[offset] for offset in token_nested_list]


def map_offsets_to_mentions(clusters, mention_map):
  def dedup_spans(cluster):
    return list(sorted(set([tuple(span) for span in cluster])))
  return [[mention_map[span] for span in dedup_spans(cluster)]
          for cluster in clusters]

def copy_mention(mention, uuid, entityType):
  return cet.EntityMention(
    uuid=uuid,
    tokens=mention.tokens,
    entityType=entityType,
    confidence=mention.confidence,
    text=mention.text)

def get_entity_type(mids, type_dict, mid_to_string):
  types = [type_dict[mid] for mid in mids]
  if not all([t[0] for t in types]):
    return (None, None)
  votes = Counter()
  for i, (t, confidence, text, start, end) in enumerate(types):
    # Break ties by insertion order
    if t.startswith("TTL"):
      confidence = min(confidence, 0.5)
    elif t.startswith("VAL"):
      confidence = max(confidence, 1.1)
    votes[t] += confidence - 0.00001 * i
  # It's hierarchical, so you actually sum supertypes into subtypes?
  # revote
  new_votes = Counter()
  keys = votes.keys()
  for key, value in votes.items():
    updatable = [k for k in keys if key in k]
    for update_key in updatable:
      new_votes[update_key] += value
  denominator = sum(list(new_votes.values()))
  entity_type, numerator = new_votes.most_common(1)[0]
  # print (new_votes)

  confidence = numerator / denominator
  return (entity_type, confidence)


def split_by_type(cluster, type_dict):
  output_clusters = []
  mentions = [m for mention in cluster for m in mention]
  types = [type_dict[m.uuid.uuidString] for m in mentions]
  vals = [[m] for t, m in zip(types, mentions) if t[0] is not None and t[0].split(".")[0] == "VAL"]
  remainder = [[m] for t, m in zip(types, mentions) if t[0] is None or t[0].split(".")[0] != "VAL"]
  # val_debug = [[t] for t, m in zip(types, mentions) if t[0].split(".")[0] == "VAL"]
  # remainder_debug = [[t] for t, m in zip(types, mentions) if t[0].split(".")[0] != "VAL"]
  # print (val_debug, remainder_debug)

  output = []
  if len(vals) > 0:
    output.append(vals)
  if len(remainder) > 0:
    output.append(remainder)
  return output

def write_span_embs(mention_clusters, span_embs, kv):
  if (len(span_embs) != len(mention_clusters)):
    print("number of embs do not match number of clusters!! embs are unreliable")
  for cluster_embs, cluster_ids in zip(span_embs, mention_clusters):
    if len(cluster_embs) != len(cluster_ids):
      print("number of embs do not match number of spans!! embs are unreliable")
    for span_emb, span_id in zip(cluster_embs, cluster_ids):
      for mention in span_id:
        kv[f"embedding[{mention.uuid.uuidString}]"] = json.dumps(span_emb)

def convert_to_comm(line_dict, examples_dict, aug, metadata, typed=False):
  not_found = 0
  skipped = 0
  if line_dict["doc_key"] not in examples_dict:
    print (f"error - preds {line_dict['doc_key']} not found in comms")
  else:
    input_example = examples_dict[line_dict["doc_key"]]
    tokenized_clusters = map_subtokens_to_tokens(line_dict["predicted_clusters"], line_dict["subtoken_map"])
    mention_clusters = map_offsets_to_mentions(tokenized_clusters, input_example[0]["mention_map"])
    comm = input_example[1]
    entityList = []
    mentionDict = {}
    seen_mention_offsets = set()
    mentionTypeDict = {}
    kv_map = {}

    if "span_embs" in line_dict:
      write_span_embs(mention_clusters, line_dict["span_embs"], kv_map)

    # Gather the types from mentions, as a type preprocessing step
    for ms_idx, mention_set in enumerate(comm.entityMentionSetList):
      # print ("{} mention_list: {}".format(ms_idx, len(mention_set.mentionList)))
      for m in mention_set.mentionList:
        tokens = m.tokens.tokenization.tokenList.tokenList
        start = tokens[m.tokens.tokenIndexList[0]].textSpan.start
        end = tokens[m.tokens.tokenIndexList[-1]].textSpan.ending
        mentionTypeDict[m.uuid.uuidString] = (m.entityType, m.confidence, m.text, start, end)
    # print (line_dict["doc_key"])

    # Create clusters
    for cluster_idx, cluster in enumerate(mention_clusters):
      cluster_parts = split_by_type(cluster, mentionTypeDict)
      for cluster_part in cluster_parts:
        possibleIM = {m.uuid.uuidString: m.entityType for mention in cluster_part for m in mention
                      if m.entityType is not None}
        picked_mention = False
        # First add mentions to mentiondict
        num_IMs = 0
        for mention in cluster_part:
          for m in mention:
            if m.uuid.uuidString in mentionDict:
              print ("Error: same mention should not be in multiple clusters")
            entityType=None
            if not picked_mention:
              if len(possibleIM) == 0 or m.entityType is not None:
                entityType="aida:informativeJustification"
                picked_mention = True
            if entityType is None or not typed:
              mentionDict[m.uuid.uuidString] = (m.uuid, None)
            else:
              num_IMs += 1
              mentionDict[m.uuid.uuidString] = (next(aug), entityType)

        original_mids = [m.uuid.uuidString for mention in cluster_part for m in mention]
        mentionIds = [mentionDict[mid][0] for mid in original_mids]

        mid_to_string = {m.uuid.uuidString: m.text for mention in cluster_part for m in mention}
        entity_type, confidence = get_entity_type(original_mids, mentionTypeDict, mid_to_string)
        # print (entity_type, confidence, [mid_to_string[m] for m in original_mids])
        # Add mentions to entity
        entity = cet.Entity(mentionIdList=mentionIds, uuid=next(aug),
                            type=entity_type, confidence=confidence)
        if "cluster_embs" in line_dict:
          kv_map[f"embedding[{entity.uuid.uuidString}]"] = json.dumps(line_dict["cluster_embs"][cluster_idx])
        entityList.append(entity)

    # Update mentionSetList, including singletons where applicable
    mentionSet = []
    seenMentions = set()
    # Make first pass over clusters
    for mention in comm.entityMentionSetList[0].mentionList:
      if mention.uuid.uuidString in mentionDict:
        start_offset = min(mention.tokens.tokenIndexList)
        end_offset = max(mention.tokens.tokenIndexList)
        start_char = mention.tokens.tokenization.tokenList.tokenList[start_offset].textSpan.start
        end_char = mention.tokens.tokenization.tokenList.tokenList[end_offset].textSpan.ending
        seenMentions.add((start_char, end_char))
        (new_uuid, entityType) = mentionDict[mention.uuid.uuidString]
        mentionSet.append(copy_mention(mention, new_uuid, entityType))

    skip_string = []
    seenMentionDict = {}
    for mention in comm.entityMentionSetList[0].mentionList:
      if mention.uuid.uuidString not in mentionDict:
        start_offset = min(mention.tokens.tokenIndexList)
        end_offset = max(mention.tokens.tokenIndexList)
        start_char = mention.tokens.tokenization.tokenList.tokenList[start_offset].textSpan.start
        end_char = mention.tokens.tokenization.tokenList.tokenList[end_offset].textSpan.ending
        skip_string.append(f"[{start_char}, {end_char}]")
        if (start_char, end_char) in seenMentions:
          skipped += 1
          skip_string.append(f"[{start_char}, {end_char}] (duplicate singleton)")
          seenMentionDict[(start_char, end_char)].mentionIdList.append(mention.uuid)
          continue
        seenMentions.add((start_char, end_char))
        new_uuid = next(aug)
        entityType="aida:informativeJustification"
        # Make new entity
        if typed:
          mentionSet.append(copy_mention(mention, new_uuid, entityType))
          entity_type, confidence, text, _, _ = mentionTypeDict[mention.uuid.uuidString]
          entity = cet.Entity(mentionIdList=[new_uuid], uuid=next(aug),
                              type=entity_type, confidence=confidence)
          seenMentionDict[(start_char, end_char)] = entity
        else:
          uuid = mention.uuid
          entity = cet.Entity(mentionIdList=[uuid], uuid=next(aug))
          seenMentionDict[(start_char, end_char)] = entity

        entityList.append(entity)

    if skip_string:
      print (f"{len(skip_string)} singletons found in {line_dict['doc_key']}: " + ", ".join(skip_string))

    if typed:
      newMentionSet = cet.EntityMentionSet(uuid=next(aug),
                                           metadata=metadata(),
                                           mentionList=mentionSet)
      comm.entityMentionSetList.append(newMentionSet)

    # Finally, create entity set
    entitySet = cet.EntitySet(entityList=entityList,
                              uuid=next(aug),
                              metadata=metadata())
    if comm.entitySetList is not None:
      comm.entitySetList.append(entitySet)
    else:
      comm.entitySetList = [entitySet]

    # Output
    if comm.keyValueMap is None:
      comm.keyValueMap = kv_map
    else:
      comm.keyValueMap.update(kv_map)
    validate_communication(comm)
    return comm, not_found, skipped


if __name__ == "__main__":
  input_comm_path = sys.argv[1]
  predictions = sys.argv[2]
  output_comm_path = sys.argv[3]
  encoder = "xlmr"
  threshold = int(sys.argv[4])
  typed = json.loads(sys.argv[5])
  entity_tool = sys.argv[6] if len(sys.argv) > 6 else None
  examples_iter = reader.make_data_iter(input_comm_path, entity_tool)
  examples_dict = {example[0]["doc_id"]: example for example in examples_iter}
  predictions = open(predictions, 'r')
  output_comms = []
  augf = AnalyticUUIDGeneratorFactory()
  aug = augf.create()
  argument_tags = f"{encoder}_{threshold}"
  model_name = "incremental_coref_v1"
  metadata = lambda: cmt.AnnotationMetadata(tool=f"jhu_{model_name}:{argument_tags}",
                                            timestamp=now_timestamp())
  notfound = 0
  skipped = 0

  for line in predictions:
    line_dict = json.loads(line)
    pred_comm, doc_key, not_found_in_ex, skipped_in_ex = convert_to_comm(line_dict, examples_dict, aug, metadata, typed=typed)
    notfound += not_found_in_ex
    skipped += skipped_in_ex
    output_comms.append((line_dict["doc_key"], pred_comm))
  print (f"Not found: {notfound}. Skipped: {skipped}")
  for doc_key, comm in output_comms:
    output_name = f"{output_comm_path}/{doc_key}.comm"
    write_communication_to_file(comm, output_name)
