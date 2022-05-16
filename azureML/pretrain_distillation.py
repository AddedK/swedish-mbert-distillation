# This code is used for task-agnostic distillation
# Some of the code is based on the tutorial code provided in TextBrewer and Hugging Face
# NOTE: There are some hard-coded details, such as the dataset paths, that you should change.

import os
import argparse
import torch
from azureml.core import Run
from torch.utils.data import DataLoader

import datasets
from transformers import AutoTokenizer
from transformers import DataCollatorForLanguageModeling
from transformers import TrainingArguments, Trainer, AutoConfig

import textbrewer 
from textbrewer import GeneralDistiller
from textbrewer import TrainingConfig, DistillationConfig
from transformers import BertConfig, AutoModelForMaskedLM
from transformers import BertConfig, AdamW
from transformers import get_linear_schedule_with_warmup
import numpy as np




run = Run.get_context()

if __name__ == "__main__":
    
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
    parser.add_argument(
        '--adapted_path',
        type=str,
        help='Path to the mbert adapted weight'
    )

    args = parser.parse_args()
    SEED = 100 
    torch.manual_seed(SEED) 
    np.random.seed(SEED)

    device='cuda' if torch.cuda.is_available() else 'cpu'

    print("===== DATA =====")
    print("DATA PATH: " + args.data_path)
    print("LIST FILES IN DATA PATH...")
    print(os.listdir(args.data_path))
    print("================")

    # Prepare dataset
    dataset_directory = "{0}/mlm_GIGA_twentyten_100_MBERT_tokenized_labeled_grouped_512_data".format(args.data_path)
    lm_dataset = datasets.load_from_disk(dataset_directory)
    lm_dataset.set_format("torch")
    print(lm_dataset)

    lm_dataset = lm_dataset.remove_columns("token_type_ids")
    print(lm_dataset)

    # Choose teacher
    # TEACHER_MODEL_NAME = "KBBERT_12" 
    # TEACHER_MODEL_NAME = "MBBERT_12" 
    TEACHER_MODEL_NAME = "MBERT_ADAPT"
    # TEACHER_MODEL_NAME = "MBERT_SUPER_ADAPT"

    
    print(args.config_path)
    TEACHER_CONFIG_FILE = "{0}/mbert_lm_teacher_config.json".format(args.config_path) 
    print(TEACHER_CONFIG_FILE)

    print("LIST FILES IN CONFIG PATH...")
    print(os.listdir(args.config_path))

    print("LIST FILES IN adapt PATH...")
    print(os.listdir(args.adapted_path))
    TEACHER_STATE_FILE =   "{0}/mbert_adapted_directory/mlm_GIGA_1990_100_MBERT_ADAPT_random.pt".format(args.adapted_path)

    
    # This is used for grouping the text together
    block_size = 512

    TRUNCATION = True # Affects how model is initialized
    if TRUNCATION:
        SUFFIX = "truncated"
    else:
        SUFFIX = "random"

    DOING_VALIDATION = False
    SMALL_MODE = False

    if SMALL_MODE:
        NUM_LAYERS_TEACHER = 6
        NUM_LAYERS_STUDENT = 3
        SIZE_STRING = "SMALL"
    else:
        NUM_LAYERS_TEACHER = 12
        NUM_LAYERS_STUDENT = 6
        SIZE_STRING = "FULL"

    # STUDENT_MODEL_NAME = "KBBERT_6" 
    # STUDENT_MODEL_NAME = "MBERT_6"  
    STUDENT_MODEL_NAME = "MBERT_6_ADAPT"  
    TOKEN_MODEL = STUDENT_MODEL_NAME.split("_")[0]


    OUTPUT_DIR = "outputs/distilled_GIGA_{0}_{1}_{2}_{3}".format(SIZE_STRING,SEED,STUDENT_MODEL_NAME,SUFFIX)
    LOG_DIR = "outputs/distilled_GIGA_{0}_{1}_{2}_{3}_logs".format(SIZE_STRING,SEED,STUDENT_MODEL_NAME,SUFFIX)
    MODEL_SAVE_FILE = "outputs/distilled_GIGA_{0}_{1}_{2}_{3}.pt".format(SIZE_STRING,SEED,STUDENT_MODEL_NAME,SUFFIX)


    print("SEED = {0}, STUDENT_MODEL_NAME = {1}, TOKEN_MODEL = {2}".format(SEED,STUDENT_MODEL_NAME, TOKEN_MODEL))
    print("OUTPUT_DIR = {0}".format(OUTPUT_DIR))


    ## Training arguments
    NUM_EPOCHS = 1
    WEIGHT_DECAY = 0.01
    LEARNING_RATE = 1e-4
    BATCH_SIZE = 4


    INTERMEDIATE_MATCHES = [    
            {'layer_T':NUM_LAYERS_TEACHER-1, 'layer_S':NUM_LAYERS_STUDENT-1, 'feature':'hidden', 'loss': 'hidden_mse','weight' : 1},
            {'layer_T':NUM_LAYERS_TEACHER-1, 'layer_S':NUM_LAYERS_STUDENT-1, 'feature':'attention', 'loss': 'attention_mse', 'weight' : 1}]

    print(INTERMEDIATE_MATCHES)
    HARD_LABEL_WEIGHT = 0
    KD_LOSS_WEIGHT = 0
    TEMPERATURE = 4 # Doesn't matter, logit loss is ignored anyway


   
    # Choose tokenizer
    if STUDENT_MODEL_NAME == "KBBERT_6":
        model_checkpoint = "KB/bert-base-swedish-cased"
    elif STUDENT_MODEL_NAME == "MBERT_6" or STUDENT_MODEL_NAME == "MBERT_6_ADAPT":
        model_checkpoint = "bert-base-multilingual-cased"
    else:
        raise Exception("Undefined STUDENT_MODEL_NAME {0}".format(STUDENT_MODEL_NAME))

    
    tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm_probability=0.15)

    ## Load teacher, modify configs
    teacher_config = BertConfig.from_json_file(TEACHER_CONFIG_FILE)
    teacher_config.output_hidden_states = True
    teacher_config.output_attentions = True
    teacher_config.num_hidden_layers = NUM_LAYERS_TEACHER
    if SMALL_MODE:
        teacher_model = AutoModelForMaskedLM.from_pretrained(TEACHER_STATE_FILE,config=teacher_config)
    else:
        teacher_model = AutoModelForMaskedLM.from_config(teacher_config) 
        teacher_model.load_state_dict(torch.load(TEACHER_STATE_FILE))
    teacher_model.to(device=device)

    # Prepare student model
    student_config = BertConfig.from_json_file(TEACHER_CONFIG_FILE) 
    student_config.output_hidden_states = True
    student_config.output_attentions = True
    student_config.num_hidden_layers= NUM_LAYERS_STUDENT # Half as many layers

    
    # Student model
    if not TRUNCATION:
        # Since we are doing model constructor and not from_pretrained, student is randomly initialized
        student_model = AutoModelForMaskedLM.from_config(student_config) 
    else:
        # Help from https://github.com/huggingface/transformers/issues/1206
        student_model = AutoModelForMaskedLM.from_pretrained(TEACHER_STATE_FILE,config=student_config)
    student_model.to(device=device)

    if DOING_VALIDATION:
        print("Doing validation, fixing datasets and dataloader")
        # It should be fixed seed irregardles of seed for model, since every model should have same data/val split
        lm_data_train = lm_dataset["train"].select(range(20000)) 
        lm_data_val = lm_dataset["train"].select(range(20000,30000))
        train_dataloader = DataLoader(lm_data_train, shuffle=True, batch_size=BATCH_SIZE, collate_fn = data_collator) 
        eval_dataloader = DataLoader(lm_data_val, shuffle=False, batch_size=BATCH_SIZE, collate_fn = data_collator)
        print(lm_data_train)
        print(lm_data_val)
    else:
        train_dataloader = DataLoader(lm_dataset["train"], shuffle=True, batch_size=BATCH_SIZE, collate_fn = data_collator) 


    if DOING_VALIDATION:
        print("Fixing predict function for validation")
        def predict(model, eval_dataset , device, step): # model=self.model_S, step = global_step mandatory keywords
            # Help from https://huggingface.co/course/chapter7/2?fw=pt
            print("doing prediction")
            model.eval()

            args = TrainingArguments(
                OUTPUT_DIR,
                evaluation_strategy="no",
                save_strategy="no",
                learning_rate= LEARNING_RATE,
                num_train_epochs=3,
                weight_decay=0.01,
                do_train=False,
                do_eval=True,
                per_device_eval_batch_size = BATCH_SIZE
            )

            trainer = Trainer(
                model=model,
                args=args,
                data_collator=data_collator,
                eval_dataset = eval_dataset ,
                tokenizer=tokenizer,
            )

            eval_results = trainer.evaluate()
            print(eval_results['eval_loss'])
            del trainer
            return eval_results['eval_loss']
        from functools import partial
        callback_fun = partial(predict, model=student_model, eval_dataset = lm_data_val, device=device) # fill other arguments
    else:
        callback_fun = None


    ## Start distillation
    num_training_steps = len(train_dataloader) * NUM_EPOCHS 
    print("num_training_steps = {0}".format(num_training_steps))

    optimizer = AdamW(student_model.parameters(), lr= LEARNING_RATE )
    scheduler_class = get_linear_schedule_with_warmup
    scheduler_args = {'num_warmup_steps':int(0.1*num_training_steps), 'num_training_steps':num_training_steps}


    def simple_adaptor(batch, model_outputs):
        return {'hidden': model_outputs.hidden_states, 'attention': model_outputs.attentions}
        

    distill_config = DistillationConfig( # Mostly default distillation config
        temperature=TEMPERATURE,
        temperature_scheduler='none',
        hard_label_weight= HARD_LABEL_WEIGHT, 
        hard_label_weight_scheduler='none', 
        kd_loss_type='ce', 
        kd_loss_weight = KD_LOSS_WEIGHT, 
        kd_loss_weight_scheduler='none', 
        probability_shift=False,
        is_caching_logits=False,
        intermediate_matches=INTERMEDIATE_MATCHES)

    train_config = TrainingConfig( # Mostly default training config
        gradient_accumulation_steps=1,
        ckpt_frequency=4, # Save weights 4 times per epoch
        ckpt_epoch_frequency=1, 
        ckpt_steps=None,
        log_dir=LOG_DIR,
        output_dir= OUTPUT_DIR,
        device=device,
        fp16=False,
        fp16_opt_level='O1',
        data_parallel=False,
        local_rank=-1)

    print("TRAIN_CONFIG")
    print(train_config)

    print("DISTILL_CONFIG")
    print(distill_config)

    distiller = GeneralDistiller(
        train_config=train_config, distill_config=distill_config,
        model_T=teacher_model, model_S=student_model, 
        adaptor_T=simple_adaptor, adaptor_S=simple_adaptor)


    with distiller:
        distiller.train(optimizer, train_dataloader, NUM_EPOCHS ,
        scheduler_class=scheduler_class, scheduler_args = scheduler_args,
        callback=callback_fun) 
