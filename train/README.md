### Download the InstaCities1M dataset

Format your `data` folder in this directory as follows:

```bash
data
├── instaCities1M
│   ├── 0.jpg
│   ├── 1.jpg
│   ├── ...
│   └── 1000000.jpg
├── train.json
└── eval.json
```

### Train the model

We can run the example script to train the model:

```bash
python3 run_clip.py \
    --output_dir ./StreetCLIP-roberta-finetuned \
    --model_name_or_path ./StreetCLIP-roberta \
    --remove_unused_columns=False \
    --image_column image_path \
    --caption_column caption \
    --train_file $PWD/data/train.json \
    --validation_file $PWD/data/eval.json \
    --do_train  --do_eval \
    --per_device_train_batch_size="16" \
    --per_device_eval_batch_size="16" \
    --learning_rate="1e-6" --warmup_ratio="0.6" --weight_decay="1e-4" --num_train_epochs="3" --gradient_accumulation_steps="12" --adam_beta1="0.90" --adam_beta2="0.98" \
    --overwrite_output_dir
```
