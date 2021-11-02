// file for litbank experiments
local base = import "base.jsonnet";
local data = import "data.jsonnet";
local encoders = import "encoders.jsonnet";
{

  // Set up base config
  local Qb(data_size, layers, split, prefix) = (
    base.base +
    encoders.spanbert_large +
    encoders.finetune_top(layers) +
    data.Qbcoref_split(split) +
    base.Name(prefix + "qb_" + split + "_" + layers + "_" + data_size) +
    {singleton_eval: true,
     mentions: true,
     num_train_examples: data_size,
    }
  ),

  //finetune full model by default
  qb: {
    ["qb_" + split]: Qb(null, 8, split, "") + base.Name("qb_" + split)
    for split in std.range(0, 4)
  },

  data_curve_da: {
    ["qb_curve_da_" + split + "_" + layers + "_" + num_examples + "_" + trial]: (
      Qb(num_examples, layers, split, "") +
      base.Name("qbcoref_da_" + split + "_" + layers + "_" + num_examples + "_" + trial) +
      {num_epochs: 100, patience: 10,
      load_model: true,
      })
    for layers in [0, 3, 6, 12, 25]
    for split in [0, 1, 2, 3, 4]
    for num_examples in [5, 10, 15, 30, 60, 120, 240]
    for trial in [0, 1]
  },

  data_curve_on: {
    ["qb_curve_on_" + split + "_" + layers + "_" + num_examples + "_" + trial]: (
      Qb(num_examples, layers, split, "") +
      base.Name("qbcoref_on_" + split + "_" + layers + "_" + num_examples + "_" + trial) +
      {num_epochs: 100, patience: 10, load_model: false}
    )
    for layers in [0, 3, 6, 12, 25]
    for split in [0, 1, 2, 3, 4]
    for num_examples in [5, 10, 15, 30, 60, 120, 240]
    for trial in [0, 1]
  } + {
    ["qb_curve_onb_" + split + "_" + layers + "_" + num_examples + "_" + trial]: (
      Qb(num_examples, layers, split, "") +
      encoders.spanbert_base +
      base.Name("qbcoref_onb_" + split + "_" + layers + "_" + num_examples + "_" + trial) +
      {num_epochs: 100, patience: 10, load_model: false}
    )
    for layers in [0, 3, 6, 12, 25]
    for split in [0, 1, 2, 3, 4]
    for num_examples in [5, 10, 15, 30, 60, 120, 240]
    for trial in [0, 1]
  },

  data_curve_fb: {
    ["qb_curve_fb_" + split + "_" + layers + "_" + num_examples + "_" + trial]:  (
      Qb(num_examples, layers, split, "") +
      encoders.spanbert_large_fb +
      base.Name("qbcoref_fb_" + split + "_" + layers + "_" + num_examples + "_" + trial) +
      {num_epochs: 100, patience: 10, load_model: false}
    )
    for layers in [0, 3, 6, 12, 25]
    for split in [0, 1, 2, 3, 4]
    for num_examples in [5, 10, 15, 30, 60, 120, 240]
    for trial in [0, 1]
  },

  // Not used for model transfer
  qb_curve: {
    ["qb_" + split + "_" + layers + "_" + data_size]: (Qb(data_size, layers, split, "") +
    {load_model: true, test_set: true})
    for layers in [24, 25]
    for split in std.range(0, 4)
    for data_size in [10, 25, 50, 100, 240, 320]
  },
}
