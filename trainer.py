import torch
import os

from incremental import Incremental
from inference import Predictor
import util
from cluster import ClusterList
from tqdm import tqdm
import random
import json
import logging
from transformers import AdamW
random.seed(0)

class Trainer(torch.nn.Module):
  def __init__(self, config, model, data):
    super(Trainer, self).__init__()
    self.model = model
    self.data = data
    self.num_epochs = config["num_epochs"]
    self.patience = config["patience"]
    self.max_grad_norm = config["max_grad_norm"]
    encoder_params = (param for name, param in self.model.named_parameters()
                      if "encoder" in name)
    regular_params = (param for name, param in self.model.named_parameters()
                      if "encoder" not in name)
    self.optimizer = torch.optim.Adam(regular_params,
                                      lr=config["adam_learning_rate"])
    self.encoder_optimizer = AdamW(encoder_params,
                                   lr=config["encoder_learning_rate"],
                                   weight_decay=0.01)
    self.optimizers = [self.optimizer]
    if config["finetune"]:
      self.optimizers.append(self.encoder_optimizer)

  def train(self, evaluator):
    best_f1 = -1.0
    best_epoch = -1
    for epoch in range(self.num_epochs):
      if epoch - best_epoch >= self.patience:
        logging.info(f"Ran out of patience, stopping on epoch {epoch} " +
                     f"(saved {best_epoch} with {best_f1:.3f})")
        return
      random.shuffle(self.data)
      self.data = self.data
      loss = self.train_epoch()
      logging.info(f"average training loss: {loss:.3f}")
      f1 = evaluator.evaluate()
      if f1 > best_f1:
        torch.save({'optimizer': self.optimizer.state_dict(),
                    'encoder_optimizer': self.encoder_optimizer.state_dict(),
                    'model': self.model.state_dict()},
                   config["log_path"])
        logging.info(f"Saved model with {f1:.3f} dev F1 on epoch {epoch}")
        best_f1 = f1
        best_epoch = epoch

  def train_epoch(self):
    train_iterator = tqdm(self.data)
    total_loss = []
    for document in train_iterator:
      clusters = ClusterList()
      start_idx = 0
      loss = 0.0
      segment_iter = util.get_segment_iter(document)
      for seg_id, (segment, mask, seglen) in segment_iter:
        loss += self.train_example(segment, document, clusters, start_idx, mask)
        start_idx += seglen
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
        self.model.clear_cache(clusters, start_idx)
        clusters.detach_()
      for optimizer in self.optimizers:
        optimizer.step()
        optimizer.zero_grad()

      train_iterator.set_description_str(desc=f"{loss:.4f}")
      total_loss.append(loss)
    return sum(total_loss) / len(total_loss) if total_loss else 0.0

  def train_example(self, segment, document, clusters, start_idx, mask):
    return self.model(segment, document, clusters, start_idx, mask, train=True)


if __name__ == "__main__":
  config = util.initialize_from_env()
  train_data = util.load_data(config["train_path"])
  eval_data = util.load_data(config["eval_path"])
  incremental_model = Incremental(config)
  trainer = Trainer(config, incremental_model, train_data)
  evaluator = Predictor(incremental_model, eval_data)
  if config["load_model"] == "true":
    saved_model = torch.load(config["log_path"])
    logging.info(f"Found old model at {config['log_path']}, loading instead")
    incremental_model.load_state_dict(saved_model["model"])
    trainer.optimizer.load_state_dict(saved_model["optimizer"])

  if not os.path.exists(config["log_dir"]):
    os.makedirs(config["log_dir"])
  config_path = config["log_dir"] + "/config.json"
  logging.info(f"Saved at {config_path}")
  config_f = open(config_path, 'w+')
  config_f.write(json.dumps(config, indent=4))

  # Train
  trainer.train(evaluator)
  # Write preds
  preds_file = config['log_dir']+'/preds.json'
  evaluator.write_preds(preds_file)
  logging.info(f"Wrote preds to {preds_file}")
