// Config file containing encoder information
local local_config = import "../local.jsonnet";
{
  local encoders_dir = local_config.encoders_dir, // all encoders lives here

  // Abstract encoder definition
  local Encoder(source, name, size) = {
    encoder_source: source,
    encoder_name: name,
    token_emb_size: size,
  },

  local HuggingFaceEncoder(name, size) = Encoder("HuggingFace", name, size),
  local CustomEncoder(name, size) = Encoder("local", name, size) + {
    ckpt_dir: encoders_dir + "/" + name,
  },


  spanbert_large: HuggingFaceEncoder("shtoshni/spanbert_coreference_large", 1024),
  spanbert_base: HuggingFaceEncoder("shtoshni/spanbert_coreference_base", 768),

  spanbert_large_pt: HuggingFaceEncoder("shtoshni/spanbert_coreference_large", 1024) + {
    ckpt_dir: encoders_dir + "/spanbert_large"
  },
  spanbert_base_pt: HuggingFaceEncoder("shtoshni/spanbert_coreference_base", 768) + {
    ckpt_dir: encoders_dir + "/spanbert_base"
  },

  spanbert_large_fb: HuggingFaceEncoder("SpanBERT/spanbert-large-cased", 1024),

  custom_large: CustomEncoder("checkpoint-230000", 1024),
  custom_base: CustomEncoder("checkpoint-120000", 768),
  xlmr_large: HuggingFaceEncoder("xlm-roberta-large", 1024),

  // Finetuning functions
  finetune_top(k): {finetune: {layers: k}}
}