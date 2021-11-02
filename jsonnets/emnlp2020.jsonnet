// These are mostly configs from the EMNLP paper
local base = import "base.jsonnet";
local data = import "data.jsonnet";
local encoders = import "encoders.jsonnet";
{
  // Segment models

  // Main
  local Spb_on(segment) = (
    base.base +
    encoders.spanbert_large_pt +
    data.Ontonotes(segment, true) +
    base.Name("spb_on_" + segment)
  ),
  local default_spb_on = Spb_on(512),

  // TrainParams: hyperparameter searches
  // Gold model exps
  local TrainParams(reset, gold, teacher_rate) = (
    default_spb_on +
    {
     reset_weights: reset,
     use_gold_spans: gold,
     teacher_forcing_rate: teacher_rate,
    }  +
    base.Name("spb_on" + "_" + reset + "_" + gold + "_" + teacher_rate)
  ),

  //Dropout, alpha, sample_rate
  local TrainParams2(dropout, method, neg, lr) = (
    default_spb_on +
    {
      dropout: std.parseJson(dropout),
      method: method,
      negative_sample_rate: std.parseJson(neg),
      adam_learning_rate: std.parseJson(lr),
    } + base.Name("spb_on_" + dropout + "_" + method + "_" + neg + "_" + lr)
  ),

  //lr, gradnorm
  local TrainParams3(lr, gradnorm) = (
    default_spb_on +
    {
      adam_learning_rate: std.parseJson(lr),
      max_grad_norm: gradnorm,
    } + base.Name("spb_on_" + lr + "_" + gradnorm)
  ),

  //lr, gradnorm, evict
  local TrainParamsEvict3(lr, gradnorm) = (
    default_spb_on +
    {
      adam_learning_rate: std.parseJson(lr),
      max_grad_norm: gradnorm,
    } + base.trunc_linscale(600)
    + base.Name("spb_on_" + lr + "_" + gradnorm + "_evict")
  ),

  // Eval exps (on dev)
  local Eval(train_seg, test_seg) = Spb_on(train_seg) + {
    preds_file: self.log_dir + "/" + "preds_" + test_seg + ".json"
  },

  // Experiments
  main: {
    ["spb_on_" + segment]: Spb_on(segment)
    for segment in [1, 5, 10, 128, 256, 384, 512]
  },

  main_mention_xe: {
    ["spb_on_" + segment + "_mention"]: Spb_on(segment) + {mentions: true}
    for segment in [1, 5, 10, 128, 256, 384, 512]
  },


  ablations: {
    ["spb_on_" + std.toString(reset)[0] + "_"
               + std.toString(gold)[0] + "_"
               + std.toString(tf)[0]]: TrainParams(reset, gold, tf)
    for reset in [true, false]
    for gold in [true, false]
    for tf in [0.0, 1.0]
  },

  ablations_2: {
    ["spb_on_" + dropout + "_" + method + "_" + neg + "_" + lr]: TrainParams2(dropout, method, neg, lr)
    for dropout in ["0.4"]
    for method in ["alpha"]
    for neg in ["1.0", "0.75"]
    for lr in ["2e-04", "1e-04", "5e-04", "5e-05"]
  },

  ablations_3: {
    ["spb_on_" + lr + "_" + gradnorm]: TrainParams3(lr, gradnorm)
    for lr in ["1e-04", "2e-04"]
    for gradnorm in [1, 5, 10]
  },

  ablations_3_evict: {
    ["spb_on_" + lr + "_" + gradnorm + "_evict"]: TrainParamsEvict3(lr, gradnorm)
    for lr in ["1e-04", "2e-04"]
    for gradnorm in [1, 5, 10]
  },

  // Final configuration, 5 runs
  main_trials: {
  ["spb_on_main_" + try]: Spb_on(512) + base.Name("spb_on_512_" + try)
  for try in [1, 2, 3, 4, 5]
  },

  // Dev grid for eval
  evaluation: {
    ["spb_on_" + train + "_eval_" + test]: Eval(train, test)
    for train in [1, 5, 10, 128, 256, 384, "512_3"]
    for test in [1, 10, 128, 512]
  },

  // Encoder experiments for finetuning, none of these were ultimately used
  encoders: {
    // spb_on_512_fb: Spb_on(512) + encoders.spanbert_large_fb + base.Name("spb_fb_on_512"),
    spb_on_512_ft: Spb_on(512) + base.Name("spb_on_512_ft") + encoders.finetune_top(12),

    // spb_b_fb_on_512_ft: Spb_b_on_ft(512) + encoders.spanbert_base_fb,
    // spb_b_on_512_finetune: Spb_b_on_ft(512),

    spb_b_on_512: Spb_on(512) + encoders.spanbert_base + base.Name("spb_b_on_512"),
    // spb_b_on_512_fb: Spb_on(512) + encoders.spanbert_base_fb + base.Name("spb_b_fb_on_512"),
    spb_b_on_512_reset: Spb_on(512) + encoders.spanbert_base + base.Name("spb_b_on_512") + {
       reset_weights: true,
    }
  },

  // These exps are unused, might not even work
  unused: {
    spb_on_512_evict_200: Spb_on(512) + {
      evict_fn: {
        name: "singletons",
        distance: 200,
      },
    } + base.Name("spb_on_512_evict_200"),
    spb_on_512_evict_500: Spb_on(512) + {
      evict_fn: {
        name: "singletons",
        distance: 500,
      },
    } + base.Name("spb_on_512_evict_500"),

    spb_on_512_evict_600: Spb_on(512) + base.trunc_linscale(600) + base.Name("spb_on_512_evict_600"),
  },

  // Test (only run once)
  test: {
    spb_on_512_gold: Spb_on(512) + {
        use_gold_spans: true,
    },
    spb_on_512_3_test: Spb_on(512) + base.trunc_linscale(600) + {
      preds_file: self.log_dir + "/" + "test.predictions.jsonlines",
      test_set: true
     },
    spb_on_512_3: Spb_on(512) + base.trunc_linscale(600),
    "spb_on_0.4_alpha_1.0_2e-04_test": TrainParams2("0.4", "alpha", "1.0", "2e-04") + base.trunc_linscale(600) + {
      preds_file: self.log_dir + "/" + "test.predictions.jsonlines",
      test_set: true
    },
  }
}