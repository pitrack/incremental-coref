// file for arrau experiments
local base = import "base.jsonnet";
local data = import "data.jsonnet";
local encoders = import "encoders.jsonnet";
{
  // Set up LitBank base config, not usable on its own (missing Name)
  local Arrau(data_size, layers) = (
    base.base +
    encoders.spanbert_large +
    encoders.finetune_top(layers) +
    data.Arrau +
    {singleton_eval: true,
     mentions: true,
     num_train_examples: data_size}
  ),

  data_curve_da: {
    ["arrau_curve_da_" + layers + "_" + num_examples + "_" + trial]: (
    Arrau(num_examples, layers) +
    base.Name("arrau_da_" + layers + "_" + num_examples + "_" + trial) +
    {num_epochs: 100, patience: 10, load_model: true})
    for layers in [25]
    for num_examples in [10, 20, 40, 80, 160, 335]
    for trial in [0, 1]
  },

  data_curve_on: {
    ["arrau_curve_on_" + layers + "_" + num_examples + "_" + trial]: (
    Arrau(num_examples, layers) +
    base.Name("arrau_on_" + layers + "_" + num_examples + "_" + trial) +
    {num_epochs: 100, patience: 10, load_model: false}
    )
    for layers in [25]
    for num_examples in [10, 20, 40, 80, 160, 335]
    for trial in [0, 1]
  } + {
    ["arrau_curve_onb_" + layers + "_" + num_examples + "_" + trial]: (
    Arrau(num_examples, layers) +
    encoders.spanbert_base +
    base.Name("arrau_onb_" + layers + "_" + num_examples + "_" + trial) +
    {num_epochs: 100, patience: 10, load_model: false}
    )
    for layers in [25]
    for num_examples in [10, 20, 40, 80, 160, 335]
    for trial in [0, 1]
  },

  data_curve_fb: {
    ["arrau_curve_fb_" + layers + "_" + num_examples + "_" + trial]: (
    Arrau(num_examples, layers) +
    encoders.spanbert_large_fb +
    base.Name("arrau_fb_" + layers + "_" + num_examples + "_" + trial) +
    {num_epochs: 100, patience: 10, load_model: false}
    )
    for layers in [25]
    for num_examples in [10, 20, 40, 80, 160, 335]
    for trial in [0, 1]
  }

}
