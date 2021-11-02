// file for litbank experiments
local base = import "base.jsonnet";
local data = import "data.jsonnet";
local encoders = import "encoders.jsonnet";
{
  // Set up LitBank base config, not usable on its own (missing Name)
  local Litbank(split, data_size, layers) = (
    base.base +
    encoders.spanbert_large +
    encoders.finetune_top(layers) +
    data.Litbank_split(split) +
    {singleton_eval: true,
     mentions: true,
     num_train_examples: data_size}
  ),

  data_curve_da: {
    ["litbank_curve_da_" + split + "_" + layers + "_" + num_examples + "_" + trial]: (
    Litbank(split, num_examples, layers) +
    base.Name("litbank_da_" + split + "_" + layers + "_" + num_examples + "_" + trial) +
    {num_epochs: 100, patience: 10,
    load_model: true,
    }
    )
    for split in [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    for layers in [0, 3, 6, 12, 24, 25]
    for num_examples in [5, 10, 20, 40, 80, 100]
    for trial in [0, 1]
  },

  data_curve_on: {
    ["litbank_curve_on_" + split + "_" + layers + "_" + num_examples + "_" + trial]: (
    Litbank(split, num_examples, layers) +
    base.Name("litbank_on_" + split + "_" + layers + "_" + num_examples + "_" + trial) +
    {num_epochs: 100, patience: 10, load_model: false,
    }
    )
    for split in [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    for layers in [0, 3, 6, 12, 24, 25]
    for num_examples in [5, 10, 20, 40, 80]
    for trial in [0, 1]
  } + {
    ["litbank_curve_onb_" + split + "_" + layers + "_" + num_examples + "_" + trial]: (
    Litbank(split, num_examples, layers) +
    encoders.spanbert_base +
    base.Name("litbank_onb_" + split + "_" + layers + "_" + num_examples + "_" + trial) +
    {num_epochs: 100, patience: 10, load_model: false,
    }
    )
    for split in [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    for layers in [0, 3, 6, 12, 24, 25]
    for num_examples in [5, 10, 20, 40, 80]
    for trial in [0, 1]
  },

  data_curve_fb: {
    ["litbank_curve_fb_" + split + "_" + layers + "_" + num_examples + "_" + trial]: (
    Litbank(split, num_examples, layers) +
    encoders.spanbert_large_fb +
    base.Name("litbank_fb_" + split + "_" + layers + "_" + num_examples + "_" + trial) +
    {num_epochs: 100, patience: 10, load_model: false,
    }
    )
    for split in [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    for layers in [0, 3, 6, 12, 24, 25]
    for num_examples in [5, 10, 20, 40, 80]
    for trial in [0, 1]
  },


  // Old exps, might be a subset of above.

  // Train on ontonotes, test on litbank
  local Spb_on(genre, gold) = (
    base.base +
    encoders.spanbert_large +
    data.Ontonotes(512, genre) +
    base.Name("spb_on_litbank_" + genre + "_" + gold) +
    data.Litbank +
    base.trunc_linscale(1200)) + {
      use_gold_spans: gold
    },

  // Train on ontonotes, test on litbank
  local Litbank_on(gold, split, rate) = (
    base.base +
    encoders.spanbert_large +
    data.Litbank_split(split) +
    base.trunc_linscale(1200) +
    base.Name("litbank_" + split + "_" + gold + "_" + rate)) + {
      use_gold_spans: gold,
      method: "alpha",
      teacher_forcing_rate: rate,
      adam_learning_rate: 2e-4,
      top_span_ratio: 0.30,
      max_span_width: 25,
      mentions: true,
      update_each_segment: true,
      singleton_eval: true,
    },

  on_litbank: {
   ["spb_on_litbank_" + genre + "_" + gold]:  Spb_on(genre, gold)
   for genre in [true, false]
   for gold in [true, false]
  },

  litbank: {
   ["litbank_" + split + "_" + gold + "_" + rate]:  Litbank_on(gold, split, rate)
   for gold in [true, false]
   for split in std.range(0, 9)
   for rate in [0, 1]
  },

  litbank_exps: {
    litbank_1_true_mean: Litbank_on(true, 1, 1) + {
      adam_learning_rate: 1e-03,
    },

    litbank_1_true_2: Litbank_on(true, 1, 1) + base.Name("litbank_1_true_2") +  {
      adam_learning_rate: 1e-03,
    },

    litbank_1_false: Litbank_on(false, 1, 0),
    litbank_2_false: Litbank_on(false, 2, 0),
    litbank_2_true: Litbank_on(true, 2, 0),
  }

}
