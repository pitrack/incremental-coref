// Edit with caution, treat this as an abstract class.
local local_config = import "../local.jsonnet";
local encoders = import "encoders.jsonnet";
{
  local log_root = local_config.log_root, // all logs live here

  trunc_linscale(distance): {
    evict_fn: {
      name: "trunc_linscale",
      distance: distance
    }
  },

  // Experiments -- final for EMNLP
  local base_config = {
    model_type: "incremental",
    load_model: "auto",
    top_span_ratio: 0.4,
    threshold: 0, //
    num_epochs: 50, // patience-based stopping
    use_gold_spans: false,
    teacher_forcing_rate: 0.0,
    reset_weights: false,
    memory_limit: local_config.gpu_gb,
    negative_sample_rate: 1.0, // whether to downsample negative links
    adam_learning_rate: 2e-04,
    encoder_learning_rate: 1e-5,
    max_grad_norm: 10,
    dropout: 0.4, //
    method: "alpha", //
    evict_fn: false,
    patience: 10,
    finetune: false, // default is false, alternative is {top_k: k}
    mentions: false,
    update_each_segment: false,
    singleton_eval: false,
    token_emb_size: 0, // needs to be changed
    debug_embs: false, // output embeddings?
    test_set: false, // evaluate on test set?
    seed: 67,
  },

  // Naming template
  Name(name): {
    run_name: name,
    log_dir: log_root + "/" + self.run_name,
    log_path: self.log_dir + "/" + "checkpoint.bin",
  },

  // Scorer Vars
  local torch_scorer_vars = {
    genre: {genre_emb_size: 20},
    spans: {
      output_size: 3000,
      span_width_buckets: 30,
      feature_size: 20,
    },
    pairwise: {
      speaker_buckets: 2,
      antecedent_distance_buckets: 10,
      segment_buckets: 3,
    }

  },

  base: base_config + torch_scorer_vars,
}
