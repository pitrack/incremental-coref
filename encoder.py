import torch
import transformers


class BertModel(torch.nn.Module):
  def __init__(self, config, device):
    super(BertModel, self).__init__()
    self.tokenizer = transformers.BertTokenizer.from_pretrained(config['bert_ckpt_dir'])
    self.model = transformers.BertModel.from_pretrained(config['bert_ckpt_dir'])
    self.device = device

  def forward(self, sentence, eval_mode=False):
    if eval_mode:
      self.model.eval()
    model_input = torch.tensor(self.tokenizer.encode(sentence[1:-1]), device=self.device).unsqueeze(0)
    outputs = self.model(model_input)
    final_layer = outputs[0]
    return final_layer
