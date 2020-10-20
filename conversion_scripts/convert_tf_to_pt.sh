echo  $data_dir/$1
transformers-cli convert --model_type bert --tf_checkpoint $data_dir/$1/model.max.ckpt --config $data_dir/$1/bert_config.json --pytorch_dump_output $data_dir/$1/pytorch_model.bin
python $2/convert_tf_to_pytorch.py $data_dir/$1/model.max.ckpt $data_dir/$1/torch_scorer_vars.bin
