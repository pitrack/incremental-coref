// file for litbank experiments
local base = import "base.jsonnet";
local data = import "data.jsonnet";
local encoders = import "encoders.jsonnet";
{

  // Set up Preco base config
  local Preco(data_size, layers, prefix) = (
    base.base +
    encoders.spanbert_large +
    encoders.finetune_top(layers) +
    data.Preco +
    base.Name(prefix + "preco_" + layers + "_" + data_size) +
    {singleton_eval: true,
     mentions: true,
     num_train_examples: data_size,
    }
  ),

  local Preco_fb(data_size, layers, prefix) = (
    Preco(data_size, layers, prefix) +
    encoders.spanbert_large_fb
  ),

  preco: {
    ["preco_" + layers + "_" + data_size]: Preco(data_size, layers, "")
    for layers in [24]
    for data_size in [10, 25, 50, 100, 500, 2500, 5000, 10000, 18000, 36000]
  },

  preco_ft: {
    ["spb_preco_" + layers + "_" + data_size]: Preco(data_size, layers, "spb_")
    for layers in [24]
    for data_size in [0, 10, 25, 50, 100, 500, 2500, 5000, 10000, 18000, 36000]
  },

  preco_fb: {
    ["fb_preco_" + layers + "_" + data_size]: Preco_fb(data_size, layers, "fb_")
    for layers in [24, 25]
    for data_size in [10, 25, 50, 100, 500, 2500, 5000, 10000, 18000, 36000, 36120]
  },

  preco_eval: {
    ["spb_preco_" + layers + "_" + data_size + "_eval"]: Preco(data_size, layers, "spb_") + {test_set: true}
    for layers in [24]
    for data_size in [10, 25, 50, 100, 500, 2500, 5000, 10000, 18000, 36000]
  },

  // Dev set variance
  dev_variance: {
    preco_15_epochs: (
    Preco_fb(2500, 24, "fb_") +
    base.Name("preco_fb_2500_15_epochs") +
    {num_epochs: 15, patience: 999}
    )
  },

  data_fork: {
    ["preco_fork_" + split]: (
    Preco_fb(2500, 24, "fb") +
    data.Preco_data_fork(split) +
    base.Name("preco_fork_" + split)
    )
    for split in ['0_0_0', '0_0_1', '0_0_2', '0_1_0', '0_1_1', '0_1_2', '0_2_0', '0_2_1', '0_2_2',
                  '1_0_0', '1_0_1', '1_0_2', '1_1_0', '1_1_1', '1_1_2', '1_2_0', '1_2_1', '1_2_2',
		  '2_0_0', '2_0_1', '2_0_2', '2_1_0', '2_1_1', '2_1_2', '2_2_0', '2_2_1', '2_2_2',
		  '0', '1', '2', '0_0', '0_1', '0_2', '1_0', '1_1', '1_2', '2_0', '2_1', '2_2']
  },

  data_curve_fb: {
    ["preco_curve_fb_" + layers + "_" + num_examples + "_" + trial]: (
    Preco_fb(num_examples, layers, "fb") +
    base.Name("preco_fb_" + layers + "_" + num_examples + "_" + trial) +
    {num_epochs: 100, patience: 10, load_model: false}
    )
    for layers in [0, 4, 8, 12, 16, 20, 24, 25]
    for num_examples in [5, 10, 25, 50, 100, 250, 500, 1000, 2500]
    for trial in [0, 1, 2]
  },


  data_curve_on: {
    ["preco_curve_on_" + layers + "_" + num_examples + "_" + trial]: (
    Preco(num_examples, layers, "on") +
    base.Name("preco_on_" + layers + "_" + num_examples + "_" + trial) +
    {num_epochs: 100, patience: 10, load_model: false}
    )
    for layers in [0, 4, 8, 12, 16, 20, 24, 25]
    for num_examples in [5, 10, 25, 50, 100, 250, 500, 1000, 2500, 1, 15, 150]
    for trial in [0, 1, 2]
  } + {
    ["preco_curve_onb_" + layers + "_" + num_examples + "_" + trial]: (
    Preco(num_examples, layers, "on") +
    encoders.spanbert_base +
    base.Name("preco_onb_" + layers + "_" + num_examples + "_" + trial) +
    {num_epochs: 100, patience: 10, load_model: false}
    )
    for layers in [0, 4, 8, 12, 16, 20, 24, 25]
    for num_examples in [5, 10, 25, 50, 100, 250, 500, 1000, 2500]
    for trial in [0, 1, 2]


  },

  data_curve_da: {
    ["preco_curve_da_" + layers + "_" + num_examples + "_" + trial]: (
    Preco(num_examples, layers, "da") +
    base.Name("preco_da_" + layers + "_" + num_examples + "_" + trial) +
    {num_epochs: 100, patience: 10,
    load_model: true,
    }
    )
    for layers in [0, 4, 8, 12, 16, 20, 24, 25]
    for num_examples in [5, 10, 25, 50, 100, 250, 500, 1000, 2500, 1, 15, 150]
    for trial in [0, 1, 2]
  },

  data_curve_xlmr: {
    ["preco_curve_xlmr_" + layers + "_" + num_examples + "_" + trial]: (
    Preco(num_examples, layers, "xlmr") +
    encoders.xlmr_large +
    base.Name("preco_xlmr_" + layers + "_" + num_examples + "_" + trial) +
    {num_epochs: 100, patience: 10}
    )
    for layers in [0, 4, 8, 12, 16, 20, 24, 25]
    for num_examples in [5, 10, 25, 50, 100, 250, 500, 1000, 2500]
    for trial in [0, 1, 2]
  },
}
