// file for ontonotes experiments
local base = import "base.jsonnet";
local data = import "data.jsonnet";
local encoders = import "encoders.jsonnet";
{

  local Ontonotes(data_size, layers, prefix) = (
    base.base +
    encoders.spanbert_large +
    encoders.finetune_top(layers) +
    data.Ontonotes(512, false) +
    base.Name(prefix + "on_" + layers + "_" + data_size) +
    {num_train_examples: data_size,}
  ),

  local Ontonotes_fb(data_size, layers, prefix) = (
    Ontonotes(data_size, layers, prefix) +
    encoders.spanbert_large_fb
  ),

  on: {
    ["on_" + layers + "_" + data_size]: Ontonotes(data_size, layers, "")
    for layers in [0, 24, 25]
    for data_size in [2802]
  },

  on_fb: {
    ["fb_on_" + layers + "_" + data_size]: Ontonotes_fb(data_size, layers, "on_")
    for layers in [24, 25]
    for data_size in [2802]
  },
}
