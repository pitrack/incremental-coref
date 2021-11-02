// file for multilingual experiments

local base = import "base.jsonnet";
local data = import "data.jsonnet";
local encoders = import "encoders.jsonnet";
{
  local base_ml(layers, data_size) = (
    base.base +
    encoders.xlmr_large +
    encoders.finetune_top(layers) +
    {num_train_examples: data_size}
  ),

  local Ontonotes_ml_en(data_size, layers, prefix) = (
    base_ml(layers, data_size) +
    data.Ontonotes_ml_english(false) +
    base.Name(prefix + "onml_en_" + layers + "_" + data_size)
  ),

  local Ontonotes_ml_zh(data_size, layers, prefix) = (
    base_ml(layers, data_size) +
    data.Ontonotes_ml_chinese(false) +
    base.Name(prefix + "onml_zh_" + layers + "_" + data_size)
  ),

  local Ontonotes_ml_ar(data_size, layers, prefix) = (
    base_ml(layers, data_size) +
    data.Ontonotes_ml_arabic(false) +
    base.Name(prefix + "onml_ar_" + layers + "_" + data_size)
  ),

  local Semeval_ml_ca(data_size, layers, prefix) = (
    base_ml(layers, data_size) +
    data.Semeval_ca +
    base.Name(prefix + "sem_ca_" + layers + "_" + data_size)
  ),

  local Semeval_ml_es(data_size, layers, prefix) = (
    base_ml(layers, data_size) +
    data.Semeval_es +
    base.Name(prefix + "sem_es_" + layers + "_" + data_size)
  ),

  local Semeval_ml_nl(data_size, layers, prefix) = (
    base_ml(layers, data_size) +
    data.Semeval_nl +
    base.Name(prefix + "sem_nl_" + layers + "_" + data_size)
  ),

  local Semeval_ml_it(data_size, layers, prefix) = (
    base_ml(layers, data_size) +
    data.Semeval_it +
    base.Name(prefix + "sem_it_" + layers + "_" + data_size)
  ),

  onml_baselines: {
    onml_en_baseline: Ontonotes_ml_en(2802, 25, "baseline_"),
    onml_zh_baseline: Ontonotes_ml_zh(1810, 25, "baseline_"),
    onml_ar_baseline: Ontonotes_ml_ar(359, 25, "baseline_"),
    sem_ca_baseline: Semeval_ml_ca(829, 25, "baseline_"),
    sem_es_baseline: Semeval_ml_es(875, 25, "baseline_"),
    sem_it_baseline: Semeval_ml_it(80, 25, "baseline_"),
    sem_nl_baseline: Semeval_ml_nl(145, 25, "baseline_"),
  },

  data_curve_en: {
    ["onml_curve_en_" + layers + "_" + num_examples + "_" + trial]: (
    Ontonotes_ml_en(num_examples, layers, "") +
    encoders.xlmr_large +
    base.Name("onml_en_" + layers + "_" + num_examples + "_" + trial) +
    {num_epochs: 100, patience: 5}
    )
    for layers in [0, 4, 8, 12, 16, 20, 24, 25]
    for num_examples in [5, 10, 25, 50, 100, 250, 500, 1000, 2500, 2802]
    for trial in [0, 1, 2]
  },

  // zh
  data_curve_zh: {
    ["onml_curve_zh_" + load_model + "_" + layers + "_" + num_examples + "_" + trial + (if gold then "_gold" else "")]: (
    Ontonotes_ml_zh(num_examples, layers, "") +
    base.Name("onml_zh_" + load_model + "_" + layers + "_" + num_examples + "_" + trial + (if gold then "_gold" else "")) +
    {num_epochs: 100, patience: 10, load_model: (load_model=="da"),
    mentions: (!gold), use_gold_spans: gold}
    )
    for layers in [0, 4, 6, 8, 12, 16, 20, 24, 25]
    for num_examples in [0, 5, 10, 25, 50, 100, 250, 500, 1000, 1810]
    for trial in [0, 1, 2]
    for load_model in ["da", "xlmr"]
    for gold in [true, false]
  },

  // ar
  data_curve_ar: {
    ["onml_curve_ar_" + load_model + "_" + layers + "_" + num_examples + "_" + trial + (if gold then "_gold" else "")]: (
    Ontonotes_ml_ar(num_examples, layers, "") +
    base.Name("onml_ar_" + load_model + "_" + layers + "_" + num_examples + "_" + trial + (if gold then "_gold" else "")) +
    {num_epochs: 100, patience: 10, load_model: (load_model=="da"),
    mentions: (!gold), use_gold_spans: gold}
    )
    for layers in [0, 4, 8, 12, 16, 20, 24, 25]
    for num_examples in [0, 5, 10, 20, 25, 40, 50, 80, 100, 160, 250, 359]
    for trial in [0, 1, 2]
    for load_model in ["da", "xlmr"]
    for gold in [true, false]
  },


  //Semeval
  data_curve_es: {
    ["onml_curve_es_" + load_model + "_" + layers + "_" + num_examples + "_" + trial + (if gold then "_gold" else "")]: (
    Semeval_ml_es(num_examples, layers, "") +
    base.Name("onml_es_" + load_model + "_" + layers + "_" + num_examples + "_" + trial + (if gold then "_gold" else "")) +
    {num_epochs: 100, patience: 10, load_model: (load_model=="da"),
    mentions: (!gold), use_gold_spans: gold}
    )
    for layers in [0, 4, 8, 12, 16, 20, 24, 25]
    for num_examples in [0, 10, 25, 50, 100, 250, 875]
    for trial in [0, 1, 2]
    for load_model in ["da", "xlmr"]
    for gold in [true, false]
  },

  data_curve_ca: {
    ["onml_curve_ca_" + load_model + "_" + layers + "_" + num_examples + "_" + trial + (if gold then "_gold" else "")]: (
    Semeval_ml_ca(num_examples, layers, "") +
    base.Name("onml_ca_" + load_model + "_" + layers + "_" + num_examples + "_" + trial + (if gold then "_gold" else "")) +
    {num_epochs: 100, patience: 10, load_model: (load_model=="da"),
    mentions: (!gold), use_gold_spans: gold}
    )
    for layers in [0, 4, 8, 12, 16, 20, 24, 25]
    for num_examples in [0, 10, 25, 50, 100, 250, 829]
    for trial in [0, 1, 2]
    for load_model in ["da", "xlmr"]
    for gold in [true, false]
  },

  data_curve_it: {
    ["onml_curve_it_" + load_model + "_" + layers + "_" + num_examples + "_" + trial + (if gold then "_gold" else "")]: (
    Semeval_ml_it(num_examples, layers, "") +
    base.Name("onml_it_" + load_model + "_" + layers + "_" + num_examples + "_" + trial + (if gold then "_gold" else "")) +
    {num_epochs: 100, patience: 10, load_model: (load_model=="da"),
    mentions: (!gold), use_gold_spans: gold}
    )
    for layers in [0, 4, 8, 12, 16, 20, 24, 25]
    for num_examples in [0, 10, 20, 40, 80]
    for trial in [0, 1, 2]
    for load_model in ["da", "xlmr"]
    for gold in [true, false]
  },

  data_curve_nl: {
    ["onml_curve_nl_" + load_model + "_" + layers + "_" + num_examples + "_" + trial + (if gold then "_gold" else "")]: (
    Semeval_ml_nl(num_examples, layers, "") +
    base.Name("onml_nl_" + load_model + "_" + layers + "_" + num_examples + "_" + trial + (if gold then "_gold" else "")) +
    {num_epochs: 100, patience: 10, load_model: (load_model=="da"),
    mentions: (!gold), use_gold_spans: gold}
    )
    for layers in [0, 4, 8, 12, 16, 20, 24, 25]
    for num_examples in [0, 10, 20, 40, 80, 145]
    for trial in [0, 1, 2]
    for load_model in ["da", "xlmr"]
    for gold in [true, false]
  },

  // LOME-related experiments below this
  // Train on ontonotes ML, test on onml
  local xlmr_onml(gold) = (
    base.base +
    encoders.xlmr_large +
    data.Ontonotes_ml(true) +
    base.Name("onml_" + gold) +
    base.trunc_linscale(1200)) + {
      use_gold_spans: gold
    },

  local xlmr_custom_base_onml(gold) = (
    base.base +
    encoders.custom_base +
    data.Ontonotes_ml(true) +
    base.Name("cus_onml_" + gold) +
    base.trunc_linscale(1200)) + {
      use_gold_spans: gold
    },

  local xlmr_custom_large_onml(gold) = (
    base.base +
    encoders.custom_large +
    data.Ontonotes_ml(true) +
    base.Name("cus_onml_" + gold) +
    base.trunc_linscale(1200)) + {
      use_gold_spans: gold
    },


  // Train on semeval ML, test on semeval
  local xlmr_semeval(gold) = (
    base.base +
    encoders.xlmr_large +
    data.Semeval +
    base.Name("semeval_" + gold) +
    base.trunc_linscale(1200)) + {
      use_gold_spans: gold
    },

  // Train on mixed (all), test on mixed
  local xlmr_mixed_ml(gold) = (
    base.base +
    encoders.xlmr_large +
    data.Mixed_ml +
    base.Name("mixed_ml_" + gold) +
    base.trunc_linscale(1200)) + {
      use_gold_spans: gold
    },

  // Just English for comparison
  local spb_on_en(gold) = (
    base.base +
    encoders.spanbert_large +
    data.Ontonotes(512, false) +
    base.Name("spb_on_en_" + gold) +
    base.trunc_linscale(1200)) + {
    use_gold_spans: gold
    },


  ml_exps: {
    xlmr_onml_true: xlmr_onml(true),
    xlmr_semeval_true: xlmr_semeval(true),
    xlmr_mixed_ml_true: xlmr_mixed_ml(true),
    xlmr_mixed_ml_false: xlmr_mixed_ml(false),
    spb_on_en_true: spb_on_en(true),

    xlmr_mixed_ml_ft_true: xlmr_mixed_ml(true) + base.Name("mixed_ml_ft_true") + {
      finetune: {layers: 12},
      update_each_segment: true,
    },

    xlmr_mixed_ml_ft_false: xlmr_mixed_ml(false) + base.Name("mixed_ml_ft_false") + {
      finetune: {layers: 12},
      update_each_segment: true,
    },
  },


  local Xlmr_ml_true_ft(layer) = (
    xlmr_mixed_ml(true) +
    base.Name("xlmr_ml_true_" + layer) +
    encoders.finetune_top(layer) + {
      update_each_segment: true,
    }
  ),

  finetune_sweeps: {
    ["xlmr_ml_true_" + layer]: Xlmr_ml_true_ft(layer)
    for layer in std.range(0, 25)} +
    {
    ["xlmr_ml_true_" + layer + "_dataless"]: Xlmr_ml_true_ft(layer) + data.Mixed_ml_dataless
    for layer in std.range(0, 25)
  },


  base_data_exps: {
    xrb_onml_true: xlmr_custom_large_onml(true),
    xrb_onml_false: xlmr_custom_large_onml(false),

    xr_base_onml_true: xlmr_custom_base_onml(true),
    xr_base_onml_false: xlmr_custom_base_onml(false) + {
      negative_sample_rate: 0.5,
      update_each_segment: true,
    },
  },

  // These were for internal purposes/debugging and unused in the paper
  dev: {
    xlmr_onml_true_dev: xlmr_onml(true),
    xlmr_semeval_true_dev: xlmr_semeval(true),
    xlmr_mixed_ml_true_dev: xlmr_mixed_ml(true),
    spb_on_en_true_dev: spb_on_en(true),
  } + {
  ["onml_recall_ar_xlmr_" + ratio + "_" + layers + "_" + num_examples + "_" + trial]: (
    Ontonotes_ml_ar(num_examples, layers, "") +
    encoders.xlmr_large +
    base.Name("onml_ar_on_" + ratio + "_" + layers + "_" + num_examples + "_" + trial) +
    {num_epochs: 100, patience: 10, load_model: false, top_span_ratio: ratio/10}
    )
    for ratio in [4, 8, 10, 20]
    for layers in [0, 4, 8, 12, 16, 20, 24, 25]
    for num_examples in [5, 10, 25, 50, 100, 250, 359]
    for trial in [0, 1, 2]
  },

}
