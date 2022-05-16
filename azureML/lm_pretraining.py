# This is the script that will pre-train mBERT
# Many functions have been adjusted from Hugging Face tutorial code.
# NOTE: There are some hard-coded details, such as the dataset paths, that you should change.


import os
import argparse
import torch
from azureml.core import Run


import transformers
import datasets
from transformers import AutoTokenizer
from transformers import AutoConfig, AutoModelForMaskedLM, TrainingArguments, Trainer, BertConfig
from transformers import DataCollatorForLanguageModeling
from transformers import get_linear_schedule_with_warmup
from transformers import AdamW
from torch.utils.data import DataLoader
import numpy as np




import torch
if __name__ == "__main__":

    # Parsing
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--data_path',
        type=str,
        help='Path to the training data'
    )
    parser.add_argument(
        '--config_path',
        type=str,
        help='Path to the model configs'
    )
    args = parser.parse_args()
    
    SEED = 100 # Seeds to be used: 100, 101, 102
    torch.manual_seed(SEED) 
    np.random.seed(SEED)

    block_size = 512


    MODEL_NAME = "MBERT_ADAPT"
    TOKEN_MODEL = MODEL_NAME.split("_")[0]

    OUTPUT_DIR = "outputs/mlm_GIGA_1990_{0}_{1}".format(SEED,MODEL_NAME)
    MODEL_SAVE_FILE = "outputs/mlm_GIGA_1990_{0}_{1}.pt".format(SEED,MODEL_NAME)

    TOKENIZED_LABELED_GROUPED_DATA_FILENAME = "mlm_GIGA_1990_{0}_{1}_tokenized_labeled_grouped_{2}_data".format(SEED,TOKEN_MODEL,block_size)


    print("SEED = {0}, MODEL_NAME = {1}, TOKEN_MODEL = {2}".format(SEED,MODEL_NAME, TOKEN_MODEL))


    print("===== DATA =====")
    print("DATA PATH: " + args.data_path)
    print("LIST FILES IN DATA PATH...")
    print(os.listdir(args.data_path))
    print("================")


    # Prepare dataset
    print("LIST FILES IN DATA PATH...")
    dataset_directory = "{0}/mlm_GIGA_1990_100_MBERT_tokenized_labeled_grouped_512_data".format(args.data_path)
    print(os.listdir(dataset_directory))
    lm_dataset = datasets.load_from_disk(dataset_directory)
    lm_dataset.set_format("torch")

    lm_dataset = lm_dataset.remove_columns("token_type_ids")
    print(lm_dataset)

    TEACHER_CONFIG_FILE = "{0}/mbert_lm_teacher_config.json".format(args.config_path) #<--- You need to specify this
    print(TEACHER_CONFIG_FILE)

    print("LIST FILES IN CONFIG PATH...")
    print(os.listdir(args.config_path))
    TEACHER_STATE_FILE = "{0}/mbert_lm_teacher.pt".format(args.config_path) #<--- You need to specify this


    # Training args
    NUM_EPOCHS = 1
    WEIGHT_DECAY = 0.01

    LEARNING_RATE = 1e-4 
    BATCH_SIZE = 4

    # Choose tokenizer 
    if MODEL_NAME == "KBBERT_6":
        model_checkpoint = "KB/bert-base-swedish-cased"
    elif MODEL_NAME == "MBERT_6" or MODEL_NAME == "MBERT_ADAPT":
        model_checkpoint = "bert-base-multilingual-cased"
    else:
        raise Exception("Undefined MODEL_NAME {0}".format(MODEL_NAME))

    tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)

    if MODEL_NAME == "MBERT_ADAPT":
        print("Pre-training mBERT adapted")
        config = BertConfig.from_json_file(TEACHER_CONFIG_FILE)
        model = AutoModelForMaskedLM.from_config(config) 
        model.load_state_dict(torch.load(TEACHER_STATE_FILE))
    else:
        raise Exception("Undefined MODEL_NAME {0}".format(MODEL_NAME))


    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm_probability=0.15)

    train_dataloader =  DataLoader(lm_dataset["train"], shuffle=True, batch_size=BATCH_SIZE)
    num_training_steps = len(train_dataloader) * NUM_EPOCHS
    SAVE_STEPS_FR = int(num_training_steps / 4)
    del train_dataloader

    optimizer = AdamW(model.parameters(), lr= LEARNING_RATE)
    scheduler_class = get_linear_schedule_with_warmup(optimizer,num_warmup_steps = 0.1*num_training_steps, num_training_steps = num_training_steps)

    training_args = TrainingArguments(
        OUTPUT_DIR,
        learning_rate = LEARNING_RATE,
        weight_decay = WEIGHT_DECAY,
        num_train_epochs = NUM_EPOCHS,
        per_device_train_batch_size = BATCH_SIZE, 
        per_device_eval_batch_size = BATCH_SIZE,
        save_strategy = "steps",
        save_steps = SAVE_STEPS_FR  
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=lm_dataset["train"],
        data_collator=data_collator,
        optimizers = (optimizer,scheduler_class)
    )

    trainer.train() 

    # Save model on disk
    torch.save(model.state_dict(), MODEL_SAVE_FILE)
