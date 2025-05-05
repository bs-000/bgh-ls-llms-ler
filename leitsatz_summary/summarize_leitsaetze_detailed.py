from collections import Counter

from lexrank import LexRank, STOPWORDS
import datasets
import numpy as np
import pandas as pd
import torch
import krippendorff as kd
from peft import LoraConfig, AutoPeftModelForCausalLM
from sklearn.model_selection import train_test_split
from statsmodels.stats.inter_rater import fleiss_kappa, aggregate_raters
from transformers import BitsAndBytesConfig, AutoModelForCausalLM, pipeline, AutoTokenizer
from trl import SFTConfig, SFTTrainer, setup_chat_format
from accelerate import PartialState

import os

import utils

from leitsatz_summary.summarize_legal_statements import get_ls_data, mod_mistral, \
    get_tokenizer, evaluate, model_str, original_str, fine_tune_models

from settings import random_state
from utils import extractive_str, entscheidungsgruende_str, leitsatz_str, server_path, summary_str, bertscore_str, \
    rouge_1_str, rouge_2_str, rouge_l_str, fscore_str, aktenzeichen_str, recall_str, precision_str

models = {mod_mistral: fine_tune_models[mod_mistral],
          }

directory_extension = 'leitsaetze_detailed/'
directory_extension_no_repetitions = 'no_repetitions/'
current_path = 'leitsatz_summary'
picture_path = 'pictures/' + directory_extension
model_result_path = 'results/' + directory_extension
eval_result_path = 'eval_results/' + directory_extension
eval_result_path_no_repetitions = eval_result_path + directory_extension_no_repetitions
log_history_path = 'log_history/' + directory_extension
manual_eval_dir = 'manual_eval/' + directory_extension
dataframes_dir = 'dataframes/' + directory_extension

num_epochs = 1
max_model_length = 32768
max_tokens_to_generate = 750
max_tokens_source = max_model_length - max_tokens_to_generate
final_model_str = '/final_model'
baseline_case_summarizer = 'baseline-CaseSummarizer'
baseline_lex_rank = 'baseline-LexRank'
annotator_str = 'Annotator'
length_str = 'length'
baseline_str = 'baseline'
eval_classes = ['Klasse ' + str(i) for i in range(1, 8)]
# https://huggingface.co/elenanereiss/bert-german-ler
ler_model_name = "elenanereiss/bert-german-ler"
ler = pipeline("ner", model=ler_model_name)  # , device=0)
option_sentence_length = 'sl'
option_word_length = 'wl'
option_key_phrases = 'kp'
option_legal_entities = 'le'
str_pad_token = '<pad>'
legal_entities = ['PER', 'RR', 'AN', 'LD', 'ST', 'STR', 'LDS', 'ORG', 'UN', 'INN', 'GRT', 'MRK', 'GS',
                  'VO', 'EUN', 'VS', 'VT', 'RS', 'LIT']
class_str = 'class'
keyphrase_list = ['So ist es auch hier', 'noch nicht abschließend geklärt', 'Aufgabe der bisherigen Rechtsprechung',
                  'Fortführung der bisherigen Rechtsprechung',
                  'Bestätigung der bisherigen Rechtsprechung', 'Bundesgerichtshof',
                  'Das Gericht gibt hiermit seine bisherige Rechtsprechung insoweit auf',
                  'Dies ist vorliegend jedoch nicht der Fall']


def summarize_lex_rank(complete_df, create_df, num_sentences_to_pick):
    documents = complete_df[entscheidungsgruende_str].apply(lambda x: utils.split_into_sentences(x)).values.tolist()
    lxr = LexRank(documents, stopwords=STOPWORDS['de'])

    create_df[summary_str] = create_df[entscheidungsgruende_str].apply(
        lambda x: ' '.join(lxr.get_summary(utils.split_into_sentences(x), summary_size=num_sentences_to_pick,
                                           threshold=0.1)))
    return create_df[[aktenzeichen_str, leitsatz_str, summary_str]]


def prepare_data_for_finetuning(example):
    """
    Converts one example to the dict for the chat template

    :param example: sample with leitsatz_str and entscheidungsgruende_str
    :return: the dict in correct format
    """
    return {"messages": [
        {"role": "system", "content": "Du bist ein rechtlicher Assistent. "
                                      "Schreibe einen Leitsatz zum folgenden Gerichtsurteil."},
        {"role": "user", "content": example[entscheidungsgruende_str]},
        {"role": "assistant", "content": example[leitsatz_str]}
    ]}


def load_quantized_model(model_id):
    """
    Loads the model with the given id

    :param model_id: one of the ids in models
    :return: model, tokenizer to this model_id
    """
    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )

    model = AutoModelForCausalLM.from_pretrained(
        models[model_id],
        device_map="auto",
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
        quantization_config=quantization_config
    )
    model.config.use_cache = False
    model.config.pretraining_tp = 1
    return model, get_tokenizer(model_id)


def load_model(model_id):
    """
    Loads the model with the given id

    :param model_id: one of the ids in models
    :return: model, tokenizer to this model_id
    """

    model = AutoModelForCausalLM.from_pretrained(
        models[model_id],
        device_map="auto",
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
    )
    model.config.use_cache = False
    model.config.pretraining_tp = 1
    return model, get_tokenizer(model_id)


def get_train_conf(output_dir):
    """
    Creates the sftconfig

    :param output_dir: Directory for saving checkpoints
    :return: the config
    """
    return SFTConfig(
        output_dir=output_dir,
        num_train_epochs=num_epochs,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=1,
        gradient_checkpointing=True,
        optim="paged_adamw_32bit",
        save_strategy='epoch',
        logging_strategy='epoch',
        learning_rate=2e-4,
        weight_decay=0.001,
        fp16=False,
        bf16=True,
        max_grad_norm=0.3,
        max_steps=-1,
        warmup_ratio=0.03,
        max_seq_length=max_model_length,
        lr_scheduler_type="constant",
        tf32=True,
        dataset_kwargs={
            "add_special_tokens": False,
            "append_concat_token": False,
        },
        gradient_checkpointing_kwargs={'use_reentrant': False},
        eval_packing=False,
        eval_strategy='epoch',
        per_device_eval_batch_size=1
    )


def load_extended_model(model_id, extension):
    model, tokenizer = load_model(model_id)
    new_tokens = [str_pad_token]
    if option_key_phrases in extension:
        start_tag, end_tag = get_tags(option_key_phrases)
        new_tokens.append(start_tag)
        new_tokens.append(end_tag)
    if option_sentence_length in extension:
        start_tag, end_tag = get_tags(option_sentence_length)
        new_tokens.append(start_tag)
        new_tokens.append(end_tag)
    if option_word_length in extension:
        start_tag, end_tag = get_tags(option_word_length)
        new_tokens.append(start_tag)
        new_tokens.append(end_tag)
    if option_legal_entities in extension:
        for entity in legal_entities:
            new_tokens.append(entity)
    tokenizer.add_tokens(new_tokens, special_tokens=True)
    model.resize_token_embeddings(len(tokenizer))
    return model, tokenizer


def do_finetuning_run_one(train_dataset, test_dataset, model_id):
    """
    Does the finetuning with PEFT and LORA. Also saves the log history.

    :param train_dataset: Data to train on
    :param test_dataset: Data for evaluation
    :param model_id: Model to use
    """
    model, tokenizer = load_quantized_model(model_id)
    tokenizer.pad_token = tokenizer.eos_token
    model_dir = server_path(current_path=current_path, path=model_result_path + '/' + model_id)
    sft_conf = get_train_conf(model_dir)

    peft_config = LoraConfig(
        lora_alpha=128,
        lora_dropout=0.1,
        r=256,
        bias="none",
        target_modules="all-linear",
        task_type="CAUSAL_LM",
    )
    tokenizer.pad_token = tokenizer.eos_token
    # Can't evaluate causallm while training yet:
    # https://github.com/huggingface/transformers/pull/32346
    trainer = SFTTrainer(
        model=model,
        args=sft_conf,
        train_dataset=train_dataset,
        formatting_func=formatting_inputs,
        tokenizer=tokenizer,
        peft_config=peft_config,
    )
    trainer.train()

    final_model_dir = model_dir + final_model_str
    trainer.save_model(final_model_dir)
    # Load PEFT model
    model = AutoPeftModelForCausalLM.from_pretrained(
        final_model_dir,
        torch_dtype=torch.bfloat16,
        device_map={"": 0})

    # Merge LoRA and base model and save
    merged_model = model.merge_and_unload()
    merged_model.save_pretrained(final_model_dir, safe_serialization=True,
                                 max_shard_size="2GB")

    log_hist = pd.DataFrame(trainer.state.log_history)
    utils.create_dir(current_path=current_path, directory_name=log_history_path, delete=False)
    utils.df_to_json(current_path=current_path, path=log_history_path + '/' + model_id,
                     dataframe=log_hist)

    generate_eval_texts(model_id=model_id, test_dataset=test_dataset, extension='')
    evaluate(model_id=model_id, eval_result_path=eval_result_path)
    store_train_results(model_id=model_id, extension='')


def do_finetuning_run_two(train_dataset, test_dataset, model_id, extension=''):
    """
    Does the finetuning with PEFT and LORA. Also saves the log history.

    :param train_dataset: Data to train on
    :param test_dataset: Data for evaluation
    :param model_id: Model to use
    :param extension: extension for saving file names
    """
    # adding tokens with qlora not possible: https://github.com/artidoro/qlora/issues/214
    model, tokenizer = load_extended_model(model_id, extension)
    model, tokenizer = setup_chat_format(model, tokenizer)
    model_dir = server_path(current_path=current_path,
                            path=model_result_path + '/' + extension + model_id + '_' + str(num_epochs))
    sft_conf = get_train_conf(model_dir)

    # remove training data with leitsatz length >1000 tokens
    train_dataset = train_dataset.filter(lambda example: len(tokenizer(example[leitsatz_str])[0]) <= 1000)
    modules_to_save = ["lm_head", "embed_tokens"]
    peft_config = LoraConfig(
        lora_alpha=128,
        lora_dropout=0.05,
        r=256,
        bias="none",
        target_modules="all-linear",
        modules_to_save=modules_to_save,
        task_type="CAUSAL_LM",
    )

    trainer = SFTTrainer(
        model=model,
        args=sft_conf,
        train_dataset=train_dataset,
        # formatting_func=formatting_inputs,
        tokenizer=tokenizer,
        peft_config=peft_config,
        eval_dataset=test_dataset,
    )
    trainer.train()

    final_model_dir = model_dir + final_model_str
    trainer.save_model(final_model_dir)
    # Load PEFT model
    model = AutoPeftModelForCausalLM.from_pretrained(
        final_model_dir,
        torch_dtype=torch.bfloat16,
        device_map={"": 0})

    # Merge LoRA and base model and save
    merged_model = model.merge_and_unload()
    merged_model.save_pretrained(final_model_dir, safe_serialization=True,
                                 max_shard_size="2GB")

    log_hist = pd.DataFrame(trainer.state.log_history)
    utils.create_dir(current_path=current_path, directory_name=log_history_path, delete=False)
    utils.df_to_json(current_path=current_path,
                     path=log_history_path + '/' + extension + model_id + '_' + str(num_epochs),
                     dataframe=log_hist)

    generate_eval_texts(model_id=model_id + '_' + str(num_epochs), test_dataset=test_dataset, extension=extension)
    evaluate(model_id=extension + model_id + '_' + str(num_epochs), eval_result_path=eval_result_path)
    store_train_results(model_id=model_id, extension=extension)


def format_prompts(data_to_format):
    """
    Formats the data to use in generation.

    :param data_to_format: Data where entscheidungsgruende_str can be addressed
    :return: the formatted prompts as a list
    """
    prompts = [f"### Urteil: {p}\n### Leitsatz: " for p in data_to_format[entscheidungsgruende_str]]
    return prompts


def generate_texts(dataset, model, tokenizer):
    """
    Method applies the model to the dataset. Dataset is prepared for chat prompt template (as dict).
    Run with multiple GPUs

    :param dataset: Dataset to generate texts to
    :param model: model to use
    :param tokenizer: tokenizer to use
    :return: a dataframe with columns leitsatz_str and summary_str
    """
    pipe = pipeline("text-generation", model=model, tokenizer=tokenizer, device_map='auto',
                    torch_dtype=torch.bfloat16)

    prompts_and_leits = [(tokenizer.apply_chat_template(sample["messages"][:2], tokenize=False,
                                                        add_generation_prompt=True),
                          sample["messages"][2]["content"]) for sample in dataset]

    prompts = [p for (p, _) in prompts_and_leits]
    leits = [l for (_, l) in prompts_and_leits]

    # prompts = format_prompts(dataset)
    # leits = dataset[leitsatz_str]
    # no temperature as we use greedy decoding / no randomness
    outputs = pipe(prompts,
                   eos_token_id=pipe.tokenizer.eos_token_id, max_new_tokens=max_tokens_to_generate,
                   pad_token_id=pipe.tokenizer.pad_token_id)
    outputs = [entry[0]['generated_text'][len(prompts[ind]):].strip() for ind, entry in enumerate(outputs)]

    result = pd.DataFrame({leitsatz_str: leits, summary_str: outputs})
    return result


def prepare_one_dataset(df):
    """
    Converts a dataframe to a dataset

    :param df: dataframe to prepare, must include leitsatz_str and entschgr
    :return: a prepared dataset
    """
    dataset = datasets.Dataset.from_pandas(df)
    dataset = dataset.map(prepare_data_for_finetuning, remove_columns=list(dataset.features),
                          batched=False)
    return dataset


def formatting_inputs(example):
    """
    Formatting function for the inputs during finetuning

    :param example: The data
    :return: the formatted text as a list
    """
    output_texts = []
    for i in range(len(example[leitsatz_str])):
        text = f"### Urteil: {example[entscheidungsgruende_str][i]}\n### Leitsatz: {example[leitsatz_str][i]}"
        output_texts.append(text)
    return output_texts


def generate_eval_texts(model_id, test_dataset, extension=''):
    """
    Generates the texts to evaluate the model. Saves them in a json file.

    :param model_id: The model to generate the texts. There must be a final model after fine-tuning!
    :param test_dataset: prepared dataset for generation.
    :param extension: Extension for loading the model
    """
    model_dir = server_path(current_path=current_path,
                            path=model_result_path + '/' + extension + model_id) + final_model_str
    device_string = PartialState().process_index
    model = AutoPeftModelForCausalLM.from_pretrained(
        model_dir,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        # device_map='auto',
        device_map={'': device_string},
        attn_implementation="flash_attention_2",

    )
    tokenizer = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True)
    tokenizer.padding_side = 'right'

    eval_results = generate_texts(dataset=test_dataset, model=model, tokenizer=tokenizer)

    utils.create_dir(current_path=current_path, directory_name=eval_result_path, delete=False)
    utils.df_to_json(current_path=current_path, path=eval_result_path + '/' + extension + model_id,
                     dataframe=eval_results)


def store_train_results(model_id, extension=''):
    """
    Method for saving the results of training (losses) for later plotting

    :param model_id: The name of the model
    :param extension: Extension for filenames
    """
    eval_file_path = eval_result_path + 'train_overview.json'
    train_res = pd.DataFrame()
    if utils.file_exists(current_path=current_path, path=eval_file_path):
        train_res = utils.df_from_json(current_path=current_path, path=eval_file_path)
    log_hist = utils.df_from_json(current_path=current_path,
                                  path=log_history_path + '/' + extension + model_id + '_' + str(num_epochs))
    eval_res = utils.df_from_json(current_path=current_path,
                                  path=eval_result_path + '/' + extension + model_id + '_' + str(num_epochs))

    repeat_str = 'repeating sentences'
    empty_str = 'empty sentences'
    # count repetitions
    eval_res[repeat_str] = eval_res[summary_str].apply(lambda text: detect_repetition(text))
    eval_res[empty_str] = eval_res[summary_str].apply(lambda text: text.strip() == '')

    new_row = {'model': [extension + model_id], 'epochs': [num_epochs],
               'train_loss': [log_hist[-1:]['train_loss'].values[0]],
               bertscore_str + fscore_str: [eval_res[bertscore_str + fscore_str].mean()],
               rouge_1_str + fscore_str: [eval_res[rouge_1_str + fscore_str].mean()],
               rouge_2_str + fscore_str: [eval_res[rouge_2_str + fscore_str].mean()],
               rouge_l_str + fscore_str: [eval_res[rouge_l_str + fscore_str].mean()],
               repeat_str: [len(eval_res[eval_res[repeat_str]])],
               empty_str: [len(eval_res[eval_res[empty_str]])]}
    if len(train_res) > 0:
        train_res = train_res.drop(
            train_res[(train_res['model'] == extension + model_id) & (train_res['epochs'] == num_epochs)].index)
    train_res = pd.concat([train_res, pd.DataFrame.from_dict(new_row)], ignore_index=True)
    train_res = train_res.sort_values(['model', 'epochs'])
    utils.df_to_json(current_path=current_path, path=eval_file_path, dataframe=train_res)


def apply_lex_rank(complete_df, test_df):
    """
    Applyies LexRank and saves the created leitsaetze with evaluation.

    :param complete_df: The whole data for initializing lexrank
    :param test_df: the test dataframe
    """
    res = summarize_lex_rank(complete_df=complete_df, create_df=test_df, num_sentences_to_pick=2)
    utils.create_dir(current_path=current_path, directory_name=eval_result_path, delete=False)
    utils.df_to_json(current_path=current_path, path=eval_result_path + baseline_lex_rank, dataframe=res)
    evaluate(model_id=baseline_lex_rank, eval_result_path=eval_result_path)


def create_manual_eval_files():
    """
    Creates and saves excel files for the manual evaluation, as well as json files with the raw data to map it to the
    models again.

    """
    all_res = pd.DataFrame()
    for model in list(models.keys()) + [baseline_lex_rank, baseline_case_summarizer]:
        df = utils.df_from_json(current_path=current_path, path=eval_result_path + model)
        df[model_str] = model
        all_res = pd.concat([all_res, df], ignore_index=True)

    # select cases per reviewer
    cases_per_package = 5
    reviewers = 5
    cases_per_reviewer = cases_per_package * (reviewers - 1)
    all_cases_count = (reviewers - 1) * cases_per_package * reviewers
    ls_index = all_res[leitsatz_str].unique()[
               :all_cases_count].tolist()  # first run: distribute 100 cases to 5 reviewers
    reviewers_cases = [[] for _ in range(reviewers)]
    for i in range(0, reviewers):
        new_cases = ls_index[:cases_per_reviewer]
        for j in range(1, reviewers):
            second_reviewer_index = (i + j) % reviewers
            third_reviewer_index = (i + j + 1) % reviewers
            if third_reviewer_index == i:
                third_reviewer_index = (third_reviewer_index + 1) % reviewers
            reviewers_cases[second_reviewer_index].extend(
                new_cases[(j - 1) * cases_per_package:((j - 1) * cases_per_package) + cases_per_package])
            reviewers_cases[third_reviewer_index].extend(
                new_cases[(j - 1) * cases_per_package:((j - 1) * cases_per_package) + cases_per_package])
        ls_index = ls_index[cases_per_reviewer:]
        reviewers_cases[i].extend(new_cases)

    # create files per reviewer
    for i in range(len(reviewers_cases)):
        print_df = pd.DataFrame()
        cases = reviewers_cases[i]
        for case in cases:
            current_lines = all_res[all_res[leitsatz_str] == case]
            aktenzeichen = [a for a in current_lines[aktenzeichen_str].unique() if pd.notna(a)][0]
            current_lines[aktenzeichen_str] = aktenzeichen
            current_lines = current_lines[[model_str, aktenzeichen_str, summary_str]].sample(frac=1)
            current_lines = pd.concat([pd.DataFrame.from_dict({aktenzeichen_str: [aktenzeichen],
                                                               model_str: [original_str], summary_str: [case]}),
                                       current_lines], ignore_index=True)
            print_df = pd.concat([print_df, current_lines], ignore_index=True)
        utils.create_dir(current_path=current_path, directory_name=manual_eval_dir, delete=False)
        utils.df_to_json(current_path=current_path, path=manual_eval_dir + 'Reviewer_' + str(i + 1) + '.json',
                         dataframe=print_df)
        print_df.loc[print_df[model_str] != original_str, model_str] = 'x'
        eval_classes_extended = eval_classes + ['Begründung']
        for j in eval_classes_extended:
            print_df[j] = ''
        print_df.to_excel(utils.server_path(current_path=current_path,
                                            path=manual_eval_dir + 'Reviewer_' + str(i + 1) + '.xlsx'),
                          index=False)


def get_metric_results(model_list, print_values=True):
    """
    Selects the results for the calculated metrics.

    :param print_values: Whether to print some statistics
    :return: The dataframe containing the results with the corresponding data
    """
    all_res = pd.DataFrame()
    df = pd.DataFrame()
    for model in list(models.keys()) + model_list:
        df = utils.df_from_json(current_path=current_path, path=eval_result_path + model)
        df[model_str] = model
        all_res = pd.concat([all_res, df], ignore_index=True)
        eval_methods = [rouge_1_str, rouge_2_str, rouge_l_str, bertscore_str]
        if print_values:
            print(model)
            for eval_m in eval_methods:
                print(eval_m + ': ' + str(df[eval_m + recall_str].min()) + ' ' +
                      str(df[eval_m + recall_str].mean()) + ' ' + str(df[eval_m + recall_str].max())
                      + ' ' + str(df[eval_m + recall_str].std()))
    all_res = all_res.merge(df[[aktenzeichen_str, leitsatz_str]], left_on=leitsatz_str, right_on=leitsatz_str,
                            suffixes=('old', ''))
    all_res = all_res.drop([aktenzeichen_str + 'old'], axis=1)
    return all_res


def get_mapped_annotated_data(second_run=False):
    """
    Reads the annotated files and maps it to the raw data again, to get the model names.

    :return: The annotated data with model names
    """
    extension = ''
    if second_run:
        extension = '_run_two'
    annotation_data = pd.DataFrame()
    for file in utils.list_dir_files(current_path=current_path, path=manual_eval_dir + 'evaluated' + extension):
        a_data = pd.read_excel(manual_eval_dir + 'evaluated' + extension + '/' + file)
        a_data.index = a_data.index.astype(str)
        raw_data = utils.df_from_json(current_path=current_path, path=manual_eval_dir + file.replace('.xlsx', '.json'))
        if second_run:
            joined = pd.merge(a_data, raw_data, on=summary_str, suffixes=('left', ''))
        else:
            joined = a_data.join(raw_data, lsuffix='left')
        joined = joined[[aktenzeichen_str, model_str, 'Begründung'] + eval_classes]
        joined[annotator_str] = file.replace('.xlsx', '')
        joined = joined[joined[model_str] != original_str]
        annotation_data = pd.concat([annotation_data, joined], ignore_index=True)
    annotation_data = annotation_data.replace('N', 'n')
    annotation_data = annotation_data.replace('Y', 'y')
    annotation_data = annotation_data.replace(0, 'n')
    annotation_data = annotation_data.replace('x', 'y')
    return annotation_data


def get_annotator_agreements(second_run=False):
    """
    Calculates FLeiss' Kappa and Krippendorfs Alpha and saves the files for the whole dataset and
    depending on model or class
    """
    extension = ''
    if second_run:
        extension = '_run_two'
    annotation_data = get_mapped_annotated_data(second_run)
    kappa_to_save = pd.DataFrame()
    alpha_to_save = pd.DataFrame()
    setting_str = 'Setting'
    # whole data
    model_list = [baseline_lex_rank, mod_mistral]
    if second_run:
        model_list = ['_' + mod_mistral + '_10', 'le_' + mod_mistral + '_10', baseline_lex_rank]
    kappa, alpha = kappa_and_alpha_for_data(annotation_data, classes=eval_classes,
                                            used_models=model_list, second_run=second_run)
    kappa[setting_str] = 'Whole Data'
    alpha[setting_str] = 'Whole Data'
    kappa_to_save = pd.concat([kappa_to_save, kappa], ignore_index=True)
    alpha_to_save = pd.concat([alpha_to_save, alpha], ignore_index=True)
    # split by model
    for model in model_list:
        kappa, alpha = kappa_and_alpha_for_data(annotation_data, classes=eval_classes,
                                                used_models=[model], second_run=second_run)
        kappa[setting_str] = 'Only ' + model
        alpha[setting_str] = 'Only ' + model
        kappa_to_save = pd.concat([kappa_to_save, kappa], ignore_index=True)
        alpha_to_save = pd.concat([alpha_to_save, alpha], ignore_index=True)
    # split by classes
    for class_name in eval_classes:
        kappa, alpha = kappa_and_alpha_for_data(annotation_data, classes=[class_name],
                                                used_models=model_list, second_run=second_run)
        kappa[setting_str] = 'Only ' + class_name
        alpha[setting_str] = 'Only ' + class_name
        kappa_to_save = pd.concat([kappa_to_save, kappa], ignore_index=True)
        alpha_to_save = pd.concat([alpha_to_save, alpha], ignore_index=True)
    utils.df_to_json(current_path=current_path, path=manual_eval_dir + extension + 'kappa.json',
                     dataframe=kappa_to_save)
    utils.df_to_json(current_path=current_path, path=manual_eval_dir + extension + 'alpha.json',
                     dataframe=alpha_to_save)


def kappa_and_alpha_for_data(data_to_work_on, classes, used_models, second_run=False):
    """
    Does the calculation of kappa and alpha.

    :param data_to_work_on: Data for the calculation
    :param classes: evalation classes to include
    :param used_models: models to include
    :return: kappa, alpha (two dataframes with the calculated results)
    """
    annotators = ['Reviewer_' + str(i) for i in range(1, 6)]

    extension = ''
    if second_run:
        extension = '_R2'
    kappa_res = pd.DataFrame()
    alpha_res = pd.DataFrame()
    for anot_one in annotators:
        res_dict_kappa = {annotator_str: [anot_one]}
        res_dict_alpha = {annotator_str: [anot_one]}
        for anot_two in annotators:
            cases_one = data_to_work_on[data_to_work_on[annotator_str] == anot_one + extension][
                aktenzeichen_str].unique()
            cases_two = data_to_work_on[data_to_work_on[annotator_str] == anot_two + extension][
                aktenzeichen_str].unique()
            cases_both = np.intersect1d(cases_one, cases_two, assume_unique=True)

            data = []
            for case in cases_both:
                for model in used_models:
                    for class_string in classes:
                        # iterate over all items
                        if len(data_to_work_on[(data_to_work_on[aktenzeichen_str] == case) &
                                               (data_to_work_on[model_str] == model) &
                                               (data_to_work_on[annotator_str] == anot_one + extension)][
                                   class_string]) == 0:
                            print('dsf')
                        eval_one = data_to_work_on[(data_to_work_on[aktenzeichen_str] == case) &
                                                   (data_to_work_on[model_str] == model) &
                                                   (data_to_work_on[annotator_str] == anot_one + extension)][
                            class_string].values[0]
                        eval_two = data_to_work_on[(data_to_work_on[aktenzeichen_str] == case) &
                                                   (data_to_work_on[model_str] == model) &
                                                   (data_to_work_on[annotator_str] == anot_two + extension)][
                            class_string].values[0]
                        data.append([eval_one, eval_two])
            agg_data = aggregate_raters(data)[0]
            kappa = round(fleiss_kappa(agg_data, method='fleiss'), 5)
            if pd.isna(kappa):  # if all values are of one class
                alpha = 1
                kappa = 1
            else:
                data_t = np.array(data).transpose()  # returns a list per rater
                alpha = round(kd.alpha(data_t, level_of_measurement='nominal'), 5)
            res_dict_kappa[anot_two] = [kappa]
            res_dict_alpha[anot_two] = [alpha]
        kappa_res = pd.concat([kappa_res, pd.DataFrame(res_dict_kappa)])
        alpha_res = pd.concat([alpha_res, pd.DataFrame(res_dict_alpha)])
    return kappa_res, alpha_res


def get_token_lengths(second_run_option=None):
    """
    Calculates the token lengths for the llms and number of words for the baseline models.

    :return: a dataframe with the results
    """
    all_data = get_ls_data()
    if second_run_option is not None:
        all_data = get_prepared_data(second_run_option)
    prepared_data = list(zip(all_data[aktenzeichen_str].values, format_prompts(all_data)))
    all_res = pd.DataFrame()
    for model in models:
        tokenizer = get_tokenizer(model_id=model)
        # var = all_data
        # var['ls_length'] = all_data[leitsatz_str].apply(lambda x: len(tokenizer(x).encodings[0].ids))
        length_data = [(model, az, len(tokenizer(prompt).encodings[0].ids)) for (az, prompt) in prepared_data]
        df = pd.DataFrame(length_data, columns=[model_str, aktenzeichen_str, length_str])
        all_res = pd.concat([all_res, df], ignore_index=True)
    simple_length = [(baseline_lex_rank, az, len(prompt.split(' '))) for (az, prompt) in prepared_data]
    df = pd.DataFrame(simple_length, columns=[model_str, aktenzeichen_str, length_str])
    all_res = pd.concat([all_res, df], ignore_index=True)
    return all_res


def get_eval_results(second_run=False):
    """
    Method calculates different results and correlations drom the calculated metrics, the input lenght of the data
    and the annotated classes
    """
    annotated_data = get_mapped_annotated_data(second_run)
    annotated_data = annotated_data.replace('y', 1)
    annotated_data = annotated_data.replace('n', 0)
    model_list = [baseline_lex_rank, mod_mistral]
    if second_run:
        model_list = ['_' + mod_mistral + '_10', 'le_' + mod_mistral + '_10', baseline_lex_rank]
    data_reviewers_combined = pd.DataFrame()
    for az in annotated_data[aktenzeichen_str].unique():
        for model in model_list:
            var = annotated_data[(annotated_data[aktenzeichen_str] == az) & (annotated_data[model_str] == model)]
            combined_classes = (var[eval_classes].sum() / var.shape[0]).round(0)
            combined_classes = pd.concat([combined_classes,
                                          pd.Series({aktenzeichen_str: az, model_str: model})]).to_frame()
            data_reviewers_combined = pd.concat([data_reviewers_combined, combined_classes.T], ignore_index=True)

    for cla in eval_classes:
        data_reviewers_combined[cla] = data_reviewers_combined[cla].astype(int)

    azs = []
    data_reviewers_combined['v'] = data_reviewers_combined[eval_classes].sum(axis=1)
    for az in data_reviewers_combined[aktenzeichen_str].unique():
        one_count = data_reviewers_combined[(data_reviewers_combined[aktenzeichen_str] == az) &
                                            (data_reviewers_combined[model_str] == model_list[0])]['v'].values[0]
        two_count = data_reviewers_combined[(data_reviewers_combined[aktenzeichen_str] == az) &
                                            (data_reviewers_combined[model_str] == model_list[1])]['v'].values[0]
        azs.append((az, abs(one_count - two_count)))

    metrics_results = get_metric_results(model_list=model_list, print_values=False)
    if second_run:
        lengths = get_token_lengths(second_run_option=[])
        lengths_ler = get_token_lengths(second_run_option=[option_legal_entities])
        lengths_ler = lengths_ler[lengths_ler[model_str] == mod_mistral]
        lengths_ler[model_str] = 'le_' + mod_mistral + '_10'
        lengths_mist = lengths[lengths[model_str] == mod_mistral]
        lengths_mist[model_str] = '_' + mod_mistral + '_10'
        lengths = pd.concat([lengths_mist, lengths_ler, lengths[lengths[model_str] == baseline_lex_rank]],
                            ignore_index=True)
    else:
        lengths = get_token_lengths()

    eval_metrics = [rouge_1_str + recall_str, rouge_2_str + recall_str, rouge_l_str + recall_str,
                    bertscore_str + recall_str, rouge_1_str + precision_str, rouge_2_str + precision_str,
                    rouge_l_str + precision_str,
                    bertscore_str + precision_str]

    model_combination = [[m] for m in model_list] + [model_list]

    print('Correlations between classes and length')
    for selection_models in model_combination:
        print(selection_models)
        print('Individual Reviewers')
        var = annotated_data.merge(lengths)
        var = var[var[model_str].isin(selection_models)]
        res = var[eval_classes + [length_str]].corr(method='spearman')
        print(res[length_str])
        print('Combined Reviewers')
        var = data_reviewers_combined.merge(lengths)
        var = var[var[model_str].isin(selection_models)]
        res = var[eval_classes + [length_str]].corr(method='spearman')
        print(res[length_str])

    print('Correlations between classes and metrics')
    for selection_models in model_combination:
        print(selection_models)
        print('Individual Reviewers')
        var = annotated_data.merge(metrics_results)
        var = var[var[model_str].isin(selection_models)]
        res = var[eval_classes + eval_metrics].corr(method='spearman')
        print(res)
        print('Combined Reviewers')
        var = data_reviewers_combined.merge(metrics_results)
        var = var[var[model_str].isin(selection_models)]
        res = var[eval_classes + eval_metrics].corr(method='spearman')
        print(res)

    print('Average class count per case')
    for selection_models in model_combination:
        print(selection_models)
        print('Individual Reviewers')
        var = annotated_data[annotated_data[model_str].isin(selection_models)]
        print(var[eval_classes].sum() / var.shape[0])
        print('Combined Reviewers')
        var = data_reviewers_combined[data_reviewers_combined[model_str].isin(selection_models)]
        print(var[eval_classes].sum() / var.shape[0])

    print('Correlations between length and metrics')
    for selection_models in model_combination:
        print(selection_models)
        var = metrics_results[metrics_results[model_str].isin(selection_models)]
        var = var.merge(lengths)
        print(var[eval_metrics + [length_str]].corr(method='spearman')[length_str])

    print('Average metrics per case')
    for selection_models in model_combination:
        print(selection_models)
        var = metrics_results[metrics_results[model_str].isin(selection_models)]
        for metric in eval_metrics:
            print(metric + ': ' + str(var[metric].min()) + ' (min) ' + str(var[metric].mean()) + ' (mean) '
                  + str(var[metric].max()) + ' (max) ' + str(var[metric].std()) + '(std)')


def run_one():
    data = get_ls_data()
    os.environ['CUDA_LAUNCH_BLOCKING'] = "1"  # use for fine-tuning
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:256"  # use for fine-tuning
    train, test = train_test_split(data, random_state=random_state, test_size=0.3, stratify=data[extractive_str])
    test, valid = train_test_split(test, random_state=random_state, test_size=0.5, stratify=test[extractive_str])
    train_d, valid_d, test_d = prepare_one_dataset(train), prepare_one_dataset(valid), prepare_one_dataset(test)

    for m_id in models:
        do_finetuning_run_one(train_dataset=train_d, test_dataset=valid_d, model_id=m_id)
        print('Done with model ' + m_id)
    apply_lex_rank(data, valid)
    create_manual_eval_files()
    get_annotator_agreements()
    get_eval_results()


def get_tags(text):
    return ' <' + text + '> ', ' </' + text + '> '


def insert_structure_data(text_data, options):
    paragraphs = text_data.split('\n')
    new_paragraphs = []
    for para in paragraphs:
        sents = utils.split_into_sentences(para)
        new_sents = []
        for sent in sents:
            new_sent = sent
            if option_legal_entities in options:
                end_str = 'end'
                start_str = 'start'
                entity_str = 'entity'
                results = ler(new_sent)
                start_index, end_index = 0, 0
                entity_id = ''
                entities = []
                for i in range(len(results)):
                    found_entity = results[i]
                    if found_entity[entity_str].startswith('B'):  # new entity
                        if end_index != start_index:
                            # there was an entity before
                            entities.append({start_str: start_index, end_str: end_index, entity_str: entity_id,
                                             'word': sent[start_index:end_index]})
                        start_index = found_entity[start_str]
                        entity_id = found_entity[entity_str][2:]
                    else:
                        end_index = found_entity[end_str]
                if end_index != start_index:
                    # there was an entity before
                    entities.append({start_str: start_index, end_str: end_index, entity_str: entity_id,
                                     'word': sent[start_index:end_index]})

                offset = 0
                for entity in entities:
                    start_tag, end_tag = get_tags(entity[entity_str])
                    new_sent = new_sent[:entity[start_str] + offset] + start_tag + new_sent[entity[start_str] + offset:]
                    offset += len(start_tag)
                    new_sent = new_sent[:entity[end_str] + offset] + end_tag + new_sent[entity[end_str] + offset:]
                    offset += len(end_tag)
            if option_sentence_length in options:
                length = len(sent.split())
                start_tag, end_tag = get_tags(option_sentence_length)
                new_sent += start_tag + str(length) + end_tag
            if option_word_length in options:
                lengths = [len(word) for word in sent.split()]
                start_tag, end_tag = get_tags(option_word_length)
                mean = sum(lengths) / len(lengths)
                new_sent += start_tag + str(round(mean, 2)) + end_tag
            if option_key_phrases in options:
                start_tag, end_tag = get_tags(option_key_phrases)
                for phrase in keyphrase_list:
                    # mark phrase, remember offset
                    occurance = new_sent.find(phrase)
                    while occurance != -1:
                        new_sent = new_sent[:occurance + 1] + start_tag + phrase.strip() + end_tag \
                                   + new_sent[occurance + len(phrase) - 1:]
                        occurance = new_sent.find(phrase, occurance + len(phrase) + len(start_tag) + len(end_tag))
            new_sents.append(new_sent)
        new_paragraphs.append(' '.join(new_sents))
    return '\n'.join(new_paragraphs)


def get_prepared_data(options):
    data = get_ls_data()
    if len(options) == 0:
        return data
    else:
        dataframe_name = '_'.join(options) + '_data.json'
        try:
            data = utils.df_from_json(current_path=current_path, path=dataframes_dir + dataframe_name)
        except OSError as _:
            data[entscheidungsgruende_str] = data[entscheidungsgruende_str] \
                .apply(lambda x: insert_structure_data(x, options))

            utils.create_dir(current_path=current_path, delete=False, directory_name=dataframes_dir)
            utils.df_to_json(current_path=current_path, path=dataframes_dir + dataframe_name,
                             dataframe=data)
        return data


def run_two(options):
    os.environ['CUDA_LAUNCH_BLOCKING'] = "1"  # use for fine-tuning
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:256"  # use for fine-tuning
    options.sort()
    extension = '_'.join(options) + '_'
    data = get_prepared_data(options)
    train, test = train_test_split(data, random_state=random_state, test_size=0.3, stratify=data[extractive_str])
    test, valid = train_test_split(test, random_state=random_state, test_size=0.5, stratify=test[extractive_str])
    train_d, valid_d, test_d = prepare_one_dataset(train), prepare_one_dataset(valid), prepare_one_dataset(test)
    for m_id in models:
        # change to test after final selection
        do_finetuning_run_two(train_dataset=train_d, test_dataset=valid_d, model_id=m_id, extension=extension)
        print('Done with model ' + m_id)
    create_manual_eval_files_run_two()
    get_annotator_agreements(second_run=True)
    get_eval_results(second_run=True)
    print('Done')


def detect_repetition(text):
    repetition_found = False
    # Satz Widerholungen
    sents = utils.split_into_sentences(text)
    for sent in sents:
        words = utils.preprocess_text(sent, [utils.pp_option_stopwords]).split()
        word_counts = Counter(words)
        if len([word_counts[val] for val in word_counts
                if word_counts[val] > 6 and val not in utils.sentence_marks and not utils.abbreviation_ending(val)
                   and val not in ['"', '/', '§', '1', '2', '3', '4', '5', '6', '7', '8', '9', '0']]) > 0:
            repetition_found = True
    sents_reduced = set(sents)
    if len(sents) - len(sents_reduced) > 1:  # more than one repeated sentence
        repetition_found = True
    return repetition_found


def print_df_token_lengths(df):
    for model in models:
        tokenizer = get_tokenizer(model_id=model)
        var = pd.DataFrame()
        var[entscheidungsgruende_str + length_str] = df[entscheidungsgruende_str].apply(
            lambda x: len(tokenizer(x).encodings[0].ids))
        var[leitsatz_str + length_str] = df[leitsatz_str].apply(lambda x: len(tokenizer(x).encodings[0].ids))
        print(leitsatz_str)
        print(str(var[leitsatz_str + length_str].min()) + ' ' + str(var[leitsatz_str + length_str].mean()) + ' ' +
              str(var[leitsatz_str + length_str].max()) + ' ' + str(var[leitsatz_str + length_str].std()))
        print(entscheidungsgruende_str)
        print(str(var[entscheidungsgruende_str + length_str].min()) + ' ' +
              str(var[entscheidungsgruende_str + length_str].mean()) + ' ' +
              str(var[entscheidungsgruende_str + length_str].max()) + ' ' +
              str(var[entscheidungsgruende_str + length_str].std()))


def count_ler_ls():
    data = get_ls_data()
    data[option_legal_entities] = data[leitsatz_str].apply(
        lambda text: insert_structure_data(text, [option_legal_entities]))
    for ler_tag in legal_entities:
        print(ler_tag + ' ' + str(data[option_legal_entities].apply(lambda text: text.count(ler_tag)).sum()))


def create_manual_eval_files_run_two():
    """
    Creates and saves excel files for the manual evaluation, as well as json files with the raw data to map it to the
    models again.

    """
    all_res = pd.DataFrame()
    for model in ['_' + mod_mistral + '_10', option_legal_entities + '_' + mod_mistral + '_10', baseline_lex_rank,
                  baseline_case_summarizer]:
        df = utils.df_from_json(current_path=current_path, path=eval_result_path + model)
        df[model_str] = model
        all_res = pd.concat([all_res, df], ignore_index=True)

    # select cases per reviewer
    cases_per_package = 3
    reviewers = 5
    cases_per_reviewer = cases_per_package * (reviewers - 1)
    all_cases_count = (reviewers - 1) * cases_per_package * reviewers
    ls_index = all_res[leitsatz_str].unique()[
               :all_cases_count].tolist()  # second run: distribute 60 cases to 5 reviewers
    reviewers_cases = [[] for _ in range(reviewers)]

    # lengths
    for model in ['_' + mod_mistral + '_10', option_legal_entities + '_' + mod_mistral + '_10', baseline_lex_rank]:
        model_data = all_res[(all_res[leitsatz_str].isin(ls_index)) & (all_res[model_str] == model)]
        print('Model: ' + model)
        print('Mean sentences generated: ' + str(model_data[summary_str].apply(lambda text:
                                                                               len(utils.split_into_sentences(text)))
                                                 .mean()))
        print('std sentences generated: ' + str(model_data[summary_str].apply(lambda text:
                                                                              len(utils.split_into_sentences(text)))
                                                .std()))
        print('Median sentences generated: ' + str(model_data[summary_str].apply(lambda text:
                                                                                 len(utils.split_into_sentences(text)))
                                                   .median()))
        print('Mean words generated: ' + str(model_data[summary_str].apply(lambda text: len(text.split())).mean()))

    for i in range(0, reviewers):
        new_cases = ls_index[:cases_per_reviewer]
        for j in range(1, reviewers):
            second_reviewer_index = (i + j) % reviewers
            third_reviewer_index = (i + j + 1) % reviewers
            if third_reviewer_index == i:
                third_reviewer_index = (third_reviewer_index + 1) % reviewers
            reviewers_cases[second_reviewer_index].extend(
                new_cases[(j - 1) * cases_per_package:((j - 1) * cases_per_package) + cases_per_package])
            reviewers_cases[third_reviewer_index].extend(
                new_cases[(j - 1) * cases_per_package:((j - 1) * cases_per_package) + cases_per_package])
        ls_index = ls_index[cases_per_reviewer:]
        reviewers_cases[i].extend(new_cases)

    # create files per reviewer
    for i in range(len(reviewers_cases)):
        print_df = pd.DataFrame()
        cases = reviewers_cases[i]
        for case in cases:
            current_lines = all_res[all_res[leitsatz_str] == case]
            aktenzeichen = [a for a in current_lines[aktenzeichen_str].unique() if pd.notna(a)][0]
            current_lines[aktenzeichen_str] = aktenzeichen
            current_lines = current_lines[[model_str, aktenzeichen_str, summary_str]].sample(frac=1)
            current_lines = pd.concat([pd.DataFrame.from_dict({aktenzeichen_str: [aktenzeichen],
                                                               model_str: [original_str], summary_str: [case]}),
                                       current_lines], ignore_index=True)
            print_df = pd.concat([print_df, current_lines], ignore_index=True)
        utils.create_dir(current_path=current_path, directory_name=manual_eval_dir, delete=False)
        utils.df_to_json(current_path=current_path, path=manual_eval_dir + 'Reviewer_' + str(i + 1) + '_R2.json',
                         dataframe=print_df)
        print_df.loc[print_df[model_str] != original_str, model_str] = 'x'
        eval_classes_extended = eval_classes + ['Begründung']
        for j in eval_classes_extended:
            print_df[j] = ''
        print_df.to_excel(utils.server_path(current_path=current_path,
                                            path=manual_eval_dir + 'Reviewer_' + str(i + 1) + '_R2.xlsx'),
                          index=False)


if __name__ == "__main__":
    run_one()
    count_ler_ls()

    run_options = []  # option_legal_entities]  # option_sentence_length]
    run_two(run_options)
    print('Done summarizing Leitsaetze in detail')
