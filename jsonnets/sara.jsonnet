local base = import "base.jsonnet";
local data = import "data.jsonnet";
local encoders = import "encoders.jsonnet";
{
  local Sara(data_size, layers, split, prefix) = (
    base.base +
    encoders.spanbert_large +
    encoders.finetune_top(layers) +
    data.Sara(split) +
    base.Name("sara_" + split + "_" + layers)) + {
      mentions: true,
      singleton_eval: true,
      num_train_examples: data_size,
    },

  data_curve: {
    ["sara_curve_" + load_model + "_" + split + "_" + layers + "_" + num_examples + "_" + trial + (if gold then "_gold" else "")]: (
      Sara(num_examples, layers, split, "") +
      base.Name("sara_" + load_model + "_" + split + "_" + layers + "_" + num_examples + "_" + trial + (if gold then "_gold" else "")) +
      {num_epochs: 100, patience: 10,
      load_model: (load_model=="da"),
      })
    for layers in [25]
    for split in [0, 1, 2, 3, 4, 5, 6]
    for num_examples in [0, 10, 20, 40, 80, 200]
    for trial in [0, 1]
    for load_model in ["da", "on"]
    for gold in [true, false]
  } + {
    ["sara_curve_onb_" + split + "_" + layers + "_" + num_examples + "_" + trial + (if gold then "_gold" else "")]: (
      Sara(num_examples, layers, split, "") +
      encoders.spanbert_base +
      base.Name("sara_onb_" + split + "_" + layers + "_" + num_examples + "_" + trial + (if gold then "_gold" else "")) +
      {num_epochs: 100, patience: 10,
      load_model: false,
      })
    for layers in [25]
    for split in [0, 1, 2, 3, 4, 5, 6]
    for num_examples in [0, 10, 20, 40, 80, 200]
    for trial in [0, 1]
    for gold in [true, false]
  } + {
    ["sara_curve_fb_" + split + "_" + layers + "_" + num_examples + "_" + trial + (if gold then "_gold" else "")]: (
      Sara(num_examples, layers, split, "") +
      encoders.spanbert_large_fb +
      base.Name("sara_fb_" + split + "_" + layers + "_" + num_examples + "_" + trial + (if gold then "_gold" else "")) +
      {num_epochs: 100, patience: 10,
      load_model: false,
      })
    for layers in [25]
    for split in [0, 1, 2, 3, 4, 5, 6]
    for num_examples in [0, 10, 20, 40, 80, 200]
    for trial in [0, 1]
    for gold in [true, false]
  },


}