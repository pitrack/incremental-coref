// Config file of all experiments in roughly reverse chronological order
// Might want to refactor all the jsonnets some day
local icoref = import "jsonnets/emnlp2020.jsonnet"; // EMNLP 2020
local aida = import "jsonnets/aida.jsonnet";        // LOME-related
local multilingual = import "jsonnets/multilingual.jsonnet"; //LOME-related
local litbank = import "jsonnets/litbank.jsonnet";  // transfer
local preco = import "jsonnets/preco.jsonnet";      // transfer
local qbcoref = import "jsonnets/qbcoref.jsonnet";  // transfer
local ontonotes = import "jsonnets/ontonotes.jsonnet";  // transfer
local arrau = import "jsonnets/arrau.jsonnet";  // transfer
local sara = import "jsonnets/sara.jsonnet"; //transfer

(
ontonotes.on
+ ontonotes.on_fb
)

+

// Configs for model transfer
(
litbank.litbank
+ litbank.on_litbank
+ litbank.litbank
+ litbank.data_curve_da
+ litbank.data_curve_on
+ litbank.data_curve_fb
)

+

(
preco.preco
+ preco.preco_ft
+ preco.preco_fb
+ preco.preco_eval
+ preco.dev_variance
+ preco.data_fork
+ preco.data_curve_fb
+ preco.data_curve_on
+ preco.data_curve_da
+ preco.data_curve_xlmr
)

+

(
qbcoref.qb
+ qbcoref.data_curve_da
+ qbcoref.data_curve_on
+ qbcoref.data_curve_fb
+ qbcoref.qb_curve
)

+ 

(
arrau.data_curve_da
+ arrau.data_curve_on
+ arrau.data_curve_fb
)

+ 

(
sara.data_curve
)

// Multilingual/LOME
+

(
multilingual.onml_baselines
+ multilingual.data_curve_en
+ multilingual.data_curve_zh
+ multilingual.data_curve_ar
+ multilingual.data_curve_ca
+ multilingual.data_curve_es
+ multilingual.data_curve_it
+ multilingual.data_curve_nl
+ multilingual.ml_exps
+ multilingual.finetune_sweeps
+ multilingual.base_data_exps
+ multilingual.dev
+ aida.dev
)

+

// Initial incremental coref model
(
icoref.main
+ icoref.ablations
+ icoref.ablations_2
+ icoref.ablations_3
+ icoref.ablations_3_evict
+ icoref.main_trials
+ icoref.encoders
+ icoref.evaluation
+ icoref.unused
+ icoref.main_mention_xe
)
