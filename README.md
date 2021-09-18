# ALBERT + DUMA

This is the source code of our paper 《DUMA: Reading Comprehension with Transposition Thinking》. The codes are written based on https://github.com/huggingface/transformers .

The codes are tested with pytorch 1.0.0 and python 3.6. If you want to use fp16 for training, please make sure the version is commit 33512f9 of https://github.com/NVIDIA/apex .

It is recommended to download the model, config and vocab file and replace the path in trainsformers/{modeling_albert.py, configuration_albert.py, tokenization_albert.py}.

Download the train.json, dev.json, test.json from https://github.com/nlpdata/dream/tree/master/data and save them into DATA_DIR.

To run ALBERT on DREAM dataset, the script is:
```bash
export DATA_DIR=/path/to/data
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python run_multiple_choice.py \
--do_lower_case \
--do_train \
--do_eval \
--overwrite_output \
--overwrite_cache \
--eval_all_checkpoints \
--task_name dream \
--per_gpu_eval_batch_size=10 \
--logging_steps 1 \
--max_seq_length 512 \
--model_type albert \
--model_name_or_path albert-base-v2 \
--data_dir $DATA_DIR \
--learning_rate 5e-6 \
--num_train_epochs 15 \
--output_dir albert_base_dream \
--per_gpu_train_batch_size=1 \
--gradient_accumulation_steps 1 \
--warmup_steps 100 \
--save_steps 764
```

To run ALBERT+DUMA on DREAM dataset, you should replace AlbertForMultipleChoice with AlbertDUMAForMultipleChoice.

Performance outputs of checkpoints will be saved in my_eval_results.txt .

For Albert xxlarge, please refer to the paper for parameter settings.

More details will be added in the future.
