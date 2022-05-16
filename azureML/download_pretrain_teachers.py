# This code downloads the config and weights of mBERT and saves them on disk.
# Afterward, you can upload them to an Azure datastore.

import torch
from transformers import AutoModelForMaskedLM, AutoConfig

teacher_mbert_name = "bert-base-multilingual-cased"

mbert_model = AutoModelForMaskedLM.from_pretrained(teacher_mbert_name)
mbert_config = AutoConfig.from_pretrained(teacher_mbert_name)

torch.save(mbert_model.state_dict(), 'mbert_lm_teacher.pt')
mbert_config.to_json_file("mbert_lm_teacher_config.json")
