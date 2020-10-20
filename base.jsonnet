// Edit with caution, treat this as an abstract class.
local local_config = import "local.jsonnet";
{
  local data_dir = local_config.data_dir, // all data lives here
  local log_root = local_config.log_root, // all output lives here

  // DATA
  Ontonotes(segment): {
    train_path: data_dir + "/train.english." + segment + ".jsonlines",
    eval_path: data_dir + "/dev.english." + segment + ".jsonlines",
    max_span_width: 30,
    genres: ["bc", "bn", "mz", "nw", "pt", "tc", "wb"],
  },

  Ontonotes_Test(segment): self.Ontonotes(segment) + {
    eval_path: data_dir + "/test.english." + segment + ".jsonlines",
  },

  // ENCODERS
  local encoders_dir = local_config.encoders_dir, // all data lives here
  local BertEncoder(name) = {
    encoder_type: "bert",
    encoder_name: name,
    bert_ckpt_dir: encoders_dir + "/" + self.encoder_name,
    vocab_file: self.bert_ckpt_dir + "/vocab.txt",
  },

  spanbert_large: BertEncoder("spanbert_large"),
  spanbert_base: BertEncoder("spanbert_base"),
  bert_large: BertEncoder("bert_large"),

  // Experiments -- EMNLP final
  base: {
    model_type: "incremental",
    load_model: "auto",
    top_span_ratio: 0.4,
    threshold: 0,
    num_epochs: 50, // rely on patience, typically stops within 15
    use_gold_spans: false,
    teacher_forcing_rate: 0.0,
    reset_weights: false,
    memory_limit: local_config.gpu_gb,
    negative_sample_rate: 1.0, // default is 1.0 (slower)
    adam_learning_rate: 2e-04,
    encoder_learning_rate: 5e-06,
    max_grad_norm: 10,
    dropout: 0.4, // default is now 0.4
    method: "alpha", //default is alpha-weighted
    evict_fn: false,  // replace false with the next four lines for eviction
    //{
    //    name: "trunc_linscale",
    //    distance: 600,
    //},
    patience: 5,
    finetune: false,
  },

  // Naming template
  Name(name): {
    run_name: name,
    log_dir: log_root + "/" + self.run_name,
    log_path: self.log_dir + "/" + "checkpoint.bin",
  },

}
