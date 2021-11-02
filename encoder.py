import torch
import transformers
import logging
import util

class Encoder(torch.nn.Module):
  def __init__(self, config, use_cache=True):
    """`use_cache` is only used in this file to recompute embeddings.
    """
    super(Encoder, self).__init__()
    # Figure out what kind of config it is.
    # Either is it downloadable from huggingface (and we need an accompanying local file)
    # Or we load everything from a local checkpoint
    if config["encoder_source"] == "HuggingFace":
      try:
        self.tokenizer = transformers.AutoTokenizer.from_pretrained(config["encoder_name"]) # replace with local files
      except:
        logging.info("Did not found tokenizer, using spanbert-cased")
        self.tokenizer = transformers.AutoTokenizer.from_pretrained("SpanBERT/spanbert-base-cased")
      self.model = transformers.AutoModel.from_pretrained(config["encoder_name"])
    else:
      # We only support XLMR otherwise
      custom_xlmr_dir = config["custom_encoder_dir"]
      logging.info(f"Applying custom XLMR encoder: {custom_xlmr_dir}")
      self.tokenizer = transformers.XLMRobertaTokenizer.from_pretrained(custom_xlmr_dir + "/vocab.txt")
      encoder_config = transformers.XLMRobertaConfig.from_json_file(custom_xlmr_dir + "/config.json")
      self.model = transformers.XLMRobertaModel.from_pretrained(
        custom_xlmr_dir + "/pytorch_model.bin",
        config=encoder_config
      )
    self.device = config["device"]

    # If there is a cached file, we'll want to back off to use the final layer embs there
    self.cached_embeddings = None
    if use_cache:
      try:
        self.cached_embeddings = torch.load(config["log_dir"] + "/embeddings.pt")
        logging.info(f"Found cached embeddings at {config['log_dir'] + '/embeddings.pt'}. Using them")
      except FileNotFoundError:
        pass

  def forward(self, sentence, doc_seg_id=None, eval_mode=False):
    if self.cached_embeddings is not None and doc_seg_id is not None:
      if doc_seg_id in self.cached_embeddings:
        return self.cached_embeddings[doc_seg_id].to(self.device)
      else:
        logging.info(f"Did not find {doc_seg_id} cached. Recomputing instead.")
    # The Predictor already puts the model in eval() mode, so this flag only used in cache_embeddings()
    if eval_mode:
      self.model.eval()
    model_input = torch.tensor(self.tokenizer.encode(sentence[1:-1]), device=self.device).unsqueeze(0)
    outputs = self.model(model_input)
    final_layer = outputs[0]
    return final_layer


def cache_embeddings(config):
  embedder = Encoder(config, use_cache=False)
  embedder = embedder.to(config["device"])
  cache_file = config["log_dir"] + "/embeddings.pt"
  embeddings = {}
  train_data = util.load_data(config["train_path"])
  eval_data = util.load_data(config["eval_path"])
  data_iterator = enumerate(eval_data + train_data)
  for doc_num, document in data_iterator:
    if doc_num % 200 == 99:
      logging.info(f"Cached {doc_num} documents")
    segment_iter = util.get_segment_iter(document)
    start_idx = 0
    for _, (segment, _, seglen) in segment_iter:
      final_layer = embedder(segment, eval_mode=True)
      doc_seg_id = f"{document['doc_key']}_{start_idx}"
      embeddings[doc_seg_id] = final_layer.detach().cpu()
      start_idx += seglen

  torch.save(embeddings, cache_file)
  logging.info(f"Saved {len(embeddings)} embeddings to {cache_file}")

if __name__ == "__main__":
  config = util.initialize_from_env()
  cache_embeddings(config)
