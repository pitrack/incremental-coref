import json
import logging
import os
import random
from tqdm import tqdm
import torch


from incremental import Incremental
from inference import Predictor, predict_and_score
import util
from cluster import ClusterList

class Trainer(torch.nn.Module):
  def __init__(self, config, model, data):
    super(Trainer, self).__init__()
    self.model = model
    self.data = data
    self.num_epochs = config["num_epochs"]
    self.patience = config["patience"]
    self.max_grad_norm = config["max_grad_norm"]
    self.eval_freq = config.get("eval_freq", len(self.data))
    self.num_encoder_layers = self.model.encoder.model.config.num_hidden_layers
    self.save_small_model = config.get("save_small_model", False)
    if config["finetune"]:
      self.finetune_threshold = self.num_encoder_layers - config["finetune"]["layers"]
    else:
      self.finetune_threshold = None
    # We treat the 0th layer (embeddings) as the n+1st layer of the
    # model, which means it can be finetuned if the conditions match
    self.setup_optimizers(config)
    self.segment_update = config["update_each_segment"]

  def train(self, evaluator):
    best_f1 = -1.0
    best_epoch = -1
    curr_epoch = -1
    sharding = self.eval_freq < len(self.data)
    for epoch in range(self.num_epochs):
      random.shuffle(self.data)
      full_data = self.data
      for subepoch, subdata in enumerate(util.gen_subepoch_iter(full_data, self.eval_freq)):
        epoch_tag = f'{epoch}' + (f".{subepoch}" if sharding else "")
        self.data = subdata
        loss = self.train_epoch()
        logging.info(f"average training loss: {loss:.3f}")
        with torch.no_grad():
          f1 = evaluator.evaluate()
          preds_file = config['log_dir']+f'/preds_{epoch_tag}.json'
          evaluator.write_preds(preds_file)
        curr_epoch += 1

        if f1 > best_f1:
          save_dict = {
            'optimizer': self.model_optimizer,
            'model': self.get_filtered_state_dict(self.model)}
          if self.encoder_optimizer is not None:
            save_dict["encoder_optimizer"] = self.encoder_optimizer
          if self.save_small_model:
            del save_dict["optimizer"]
            del save_dict["encoder_optimizer"]
          torch.save(save_dict, config["log_path"])
          logging.info(f"Saved model with {f1:.3f} dev F1 on epoch {epoch_tag}")
          best_f1 = f1
          best_epoch = curr_epoch

        if curr_epoch - best_epoch >= self.patience:
          logging.info(f"Ran out of patience, stopping on epoch {epoch_tag} " +
                       f"(saved {best_epoch} with {best_f1:.3f})")
          return
      self.data = full_data

  def setup_optimizers(self, config):
    encoder_param_list = []
    model_param_list = []
    for name, param in self.model.named_parameters():
      if "encoder" not in name:
        model_param_list.append(param)
      elif not self.is_unused_layer(name):
        encoder_param_list.append(param)
      else:
        param.required_grad = False

    model_params = iter(model_param_list)
    self.model_optimizer = torch.optim.Adam(model_params,
                                            lr=config["adam_learning_rate"])
    self.encoder_optimizer = None
    self.optimizers = [self.model_optimizer]
    if len(encoder_param_list) > 0:
      encoder_params = iter(encoder_param_list)
      self.encoder_optimizer = torch.optim.AdamW(encoder_params,
                                                 lr=config["encoder_learning_rate"])
      self.optimizers.append(self.encoder_optimizer)
    logging.info(f"Optimizing {len(model_param_list)} (model) and " +
                 f"{len(encoder_param_list)} (encoder) parameters")

  def is_unused_layer(self, name):
    if "encoder" not in name:
      return False # Only touch encoder
    if self.finetune_threshold is None:
      return True # Encoder is frozen if no finetuning
    if self.finetune_threshold < 0:
      return False # If finetuning this much, entire encoder is unfrozen
    if "layer" not in name:
      return True # Otherwise we care about select layers: embedder is frozen
    # Freeze lower layers
    return int(name.split(".")[4]) < self.finetune_threshold

  def get_filtered_state_dict(self, module):
    return {name: params for name, params in module.state_dict().items()
            if not self.is_unused_layer(name)}

  def step_optimizers(self):
    for optimizer in self.optimizers:
      optimizer.step()
      optimizer.zero_grad()

  def train_epoch(self):
    self.model.train()
    train_iterator = tqdm(self.data)
    total_loss = []
    for document in train_iterator:
      clusters = ClusterList()
      num_runs = 1
      for run in range(num_runs):
        clusters.reset()
        self.model.set_threshold(1 * (num_runs - 1 - run))
        start_idx = 0
        loss = 0.0
        segment_iter = util.get_segment_iter(document)
        for seg_id, (segment, mask, seglen) in segment_iter:
          loss += self.train_example(segment, document, clusters, start_idx, mask)
          start_idx += seglen
          torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
          self.model.clear_cache(clusters, start_idx)
          clusters.detach_()
          if self.segment_update:
            self.step_optimizers()
      if not self.segment_update:
        self.step_optimizers()
      train_iterator.set_description_str(desc=f"{loss:.4f}")
      total_loss.append(loss)
    return sum(total_loss) / len(total_loss) if total_loss else 0.0

  def train_example(self, segment, document, clusters, start_idx, mask):
    return self.model(segment, document, clusters, start_idx, mask, train=True, consolidate=(start_idx == 0))


if __name__ == "__main__":
  config = util.initialize_from_env()
  train_data = util.load_data(config["train_path"], config.get("num_train_examples"))
  if "samples" in config:
    random.seed(config["samples"]["seed"])
    train_data = random.sample(train_data, config["samples"]["num_samples"])
  logging.info(f"Training on {len(train_data)}")
  dev_data = util.load_data(config["dev_path"], config.get("num_dev_examples"))
  incremental_model = Incremental(config)
  trainer = Trainer(config, incremental_model, train_data)
  evaluator = Predictor(incremental_model, dev_data, config["singleton_eval"])
  if config["load_model"]:
    util.load_params(incremental_model, config["load_path"], "model")
    util.load_params(trainer.model_optimizer, config["load_path"], "optimizer")
    logging.info(f"Updating threshold to {config['threshold']}")
    incremental_model.set_threshold(config["threshold"])

  if not os.path.exists(config["log_dir"]):
    os.makedirs(config["log_dir"])
  config_path = config["log_dir"] + "/config.json"
  logging.info(f"Saved at {config_path}")
  config["device"] = str(config["device"])
  config_f = open(config_path, 'w+')
  config_f.write(json.dumps(config, indent=4))
  config_f.close()

  # set seed
  util.set_seed(config)

  # Train
  if len(train_data) > 0:
    trainer.train(evaluator)

  # Run post-training predictions
  logging.info("Now running post-training dev set evaluation - reloading best checkpoint:")
  config["load_path"] = config["log_path"]
  config["load_model"] = True
  predict_and_score(config, "dev")
  # Perform test evaluation
  if config["test_set"]:
    logging.info("Now running post-training test set evaluation - reloading best checkpoint:")
    predict_and_score(config, "test")
