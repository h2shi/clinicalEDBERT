# clinicalEDBERT
## Pretrained clinicalBERT
Download the pretrained clinicalBERT model weights and configuration using this [link](https://drive.google.com/drive/folders/1kj1mvavnWIXLNxoO35oV3PZrAVdOF7_q?usp=drive_link)
This script shows the model folder's structure:
```
- model
  - pretraining
    - bert_config.json
    - pytorch_model.bin
    - vocab.txt
```

## Hospital admission using clinicalEDBERT
Using the script below to predict hospital admission:
```
python ./run_clinicalEDBERT.py \
  --task_name readmission \
  --readmission_mode ed \
  --do_train \
  --do_eval \
  --data_dir ./(DATA FILE) \
  --bert_model ./ClinicalBERT_pretraining_pytorch_checkpoint \
  --max_seq_length 128 \
  --train_batch_size (BATCH SIZE) \
  --learning_rate 2e-5 \
  --num_train_epochs (EPOCHs) \
  --output_dir ./(OUTPUT FILE)
```
The results will be in the output_dir folder and it consists of
* 'logits_clinicalbert.csv': logits from ClinicalBERT to compare with other models
* 'auprc_clinicalbert.png': Precision-Recall Curve
* 'auroc_clinicalbert.png': ROC Curve
* 'loss_history.png': Training Loss Curve
* 'eval_results.txt': RP80, accuracy, loss

## Reference
[ClinicalBERT: Modeling Clinical Notes and Predicting Hospital Readmission](https://arxiv.org/abs/1904.05342)
```
@article{clinicalbert,
author = {Kexin Huang and Jaan Altosaar and Rajesh Ranganath},
title = {ClinicalBERT: Modeling Clinical Notes and Predicting Hospital Readmission},
year = {2019},
journal = {arXiv:1904.05342},
}
```
