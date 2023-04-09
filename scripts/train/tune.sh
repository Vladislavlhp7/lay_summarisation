#!/bin/bash
# Script to start the finetuning of the model.

set -eu

# TODO: Determine use for following variables.

# Target corpus fine-tuning
python -m laysummarisation.trainer.train \
    --ftrain ${SEED_DIR}/data/train.mrp \
    --fvalid ${SEED_DIR}/data/dev.mrp \
    --ftest ${SEED_DIR}/data/test.mrp \
    --seed ${SEED} \
    --model_name_or_path ${SEED_DIR}/pretrain/model \
    --log ${SEED_DIR}/finetune \
    --split_document false \
    --attention_window 512 \
    --batch_size 4 \
    --eval_batch_size 16 \
    --postprocessor "default:default,aaec:aaec,aaec_essay:aaec,aaec_para:aaec,cdcp:cdcp,abstrct:abstrct,trees:aaec,graph:cdcp" \
    --lambda_bio 1.0 \
    --lambda_proposition ${lambda_proposition} \
    --lambda_arc ${lambda_arc} \
    --lambda_rel ${lambda_rel} \
    --lambda_tgt_fw 1.0 \
    --lambda_other_fw 1.0 \
    --lr ${lr} \
    --beta1 0.9 \
    --beta2 0.998 \
    --warmup_ratio 0.1 \
    --clip 5.0 \
    --epochs ${finetune_epochs} \
    --terminate_epochs ${finetune_epochs} \
    --evaluate_epochs 2 \
    --disable_evaluation false
# -> The trained model was saved into "${SEED_DIR}/finetune/model"
