#!/usr/bin/env python3

import pandas as pd
from transformers import AutoTokenizer

import metrics
import utils
from custom_rouge import rouge_l, rouge_n
from data.download_rii import get_selected_bgh_data
from utils import server_path, entscheidungsgruende_str, leitsatz_str, filter_topical_leitsaetze, aktenzeichen_str, \
    extractive_str, summary_str, rouge_1_str, rouge_2_str, rouge_l_str, bertscore_str, precision_str, \
    recall_str, fscore_str

current_path = 'leitsatz_summary/'
directory_extension = 'legal_statements/'
eval_result_path_simple = 'eval_results/leitsatz_summary_simple/'
picture_path_simple = 'pictures/leitsatz_summary_simple/'
picture_path_manual_results = 'pictures/'+directory_extension
dataframes_dir = 'dataframes/' + directory_extension
manual_eval_dir = 'manual_eval/' + directory_extension
dataframe_name_original_data = 'original_data.json'

# https://huggingface.co/DiscoResearch/Llama3-DiscoLeo-Instruct-8B-v0.1
mod_llama3_short_inst = 'llama3_8k_instruct'
# https://huggingface.co/DiscoResearch/Llama3-DiscoLeo-Instruct-8B-32k-v0.1
mod_llama3_long_inst = 'llama3_32k_instruct'
# https://huggingface.co/occiglot/occiglot-7b-de-en-instruct/tree/main
mod_occiglot_inst = 'occiglot_32k_instruct'
# https://huggingface.co/LeoLM/leo-mistral-hessianai-7b-chat
mod_mistral_inst = 'mistral_32k_instruct'
# https://huggingface.co/LeoLM/leo-hessianai-7b-chat
mod_llama2_inst = 'llama2_8k_instruct'
instruct_models = {
    mod_llama3_short_inst: 'DiscoResearch/Llama3-DiscoLeo-Instruct-8B-v0.1',
    mod_llama3_long_inst: 'DiscoResearch/Llama3-DiscoLeo-Instruct-8B-32k-v0.1',
    mod_occiglot_inst: 'occiglot/occiglot-7b-de-en-instruct',
    mod_llama2_inst: 'LeoLM/leo-hessianai-7b-chat',
    mod_mistral_inst: 'LeoLM/leo-mistral-hessianai-7b-chat'
}
# https://huggingface.co/DiscoResearch/Llama3-German-8B
mod_llama3_short = 'llama3_8k'
# https://huggingface.co/DiscoResearch/Llama3-German-8B-32k
mod_llama3_long = 'llama3_32k'
# https://huggingface.co/occiglot/occiglot-7b-de-en
mod_occiglot = 'occiglot_32k'
# https://huggingface.co/LeoLM/leo-mistral-hessianai-7b
mod_mistral = 'mistral_32k'
# https://huggingface.co/LeoLM/leo-hessianai-7b
mod_llama2 = 'llama2_8k'
fine_tune_models = {
    mod_mistral: "LeoLM/leo-mistral-hessianai-7b",
    mod_llama2: 'LeoLM/leo-hessianai-13b',
    mod_llama3_long: 'DiscoResearch/Llama3-German-8B-32k',
    mod_llama3_short: 'DiscoResearch/Llama3-German-8B',
    mod_occiglot: 'occiglot/occiglot-7b-de-en'
}
mod_random = 'random_sentence'
mod_old = 'simple_old_model'
models = {**instruct_models, **fine_tune_models}
model_str = 'model'
original_str = 'Original'


def prepare_raw_data(row):
    """
    Prepares Leitsatz (removes leading listing etc) and Entscheidungsgruende (Removes Randnummern, but inserts '\n' for
    them) and determines, whether the leitsatz is extractive. For one row in parallel mode.

    :param row: dataframe row, containing aktenzeichen, leitsatz and entscheidungsgruende raw
    :return: a new dataframe with the prepared data (aktenzeichen, leitsatz, entscheidungruende, extractive?)
    """
    index, row_data = row
    ls = utils.prepare_leitsatz(row_data[leitsatz_str])
    eg = utils.prepare_entsch_gr(row_data[entscheidungsgruende_str])
    extractive = set(ls).issubset(eg)
    eg_final = ''
    for i in range(len(eg)):
        current = eg[i]
        if not current.isdigit():
            eg_final += ' ' + current
        else:
            eg_final += '\n'  # Randnummern
    eg_final = eg_final.strip()
    res = pd.DataFrame({aktenzeichen_str: [row_data[aktenzeichen_str]],
                        leitsatz_str: [' '.join(ls)],
                        entscheidungsgruende_str: [eg_final],
                        extractive_str: [extractive]})
    return res


def get_ls_data():
    """
    Method to coordinate the loading of the original data. If it is already saved in a file, the file is loaded,
    otherwise the data is prepared and saved

    :return: The data to work on. A dataframe with the columns
            (aktenzeichen, leitsatz, entscheidungsgruende, extractive)
    """
    try:
        data_res = utils.df_from_json(current_path=current_path, path=dataframes_dir + dataframe_name_original_data)
    except OSError as _:
        data_raw = get_selected_bgh_data(case=0, directory=server_path(current_path=current_path, path='../data/'))
        data_raw = data_raw.dropna(subset=[leitsatz_str, entscheidungsgruende_str])
        data_raw = filter_topical_leitsaetze(data_raw)
        data_raw = data_raw[[aktenzeichen_str, leitsatz_str, entscheidungsgruende_str]]
        results = utils.parallel_apply_async(function=prepare_raw_data, data=data_raw)
        data_res = pd.DataFrame()
        for res in results:
            data_res = pd.concat([data_res, res.get()], ignore_index=True)
        utils.create_dir(current_path=current_path, delete=False, directory_name=dataframes_dir)
        utils.df_to_json(current_path=current_path, path=dataframes_dir + dataframe_name_original_data,
                         dataframe=data_res)
    data_res = data_res.drop_duplicates(leitsatz_str)
    return data_res


def apply_rouge(func, created, reference, identifier, n=None):
    """
    Function to apply any rouge function.

    :param func: the function to apply
    :param created: the created summary
    :param reference: the gold summary
    :param identifier: the identifier of the method
    :param n: n for ROUGE-n, None for ROUGE-l
    :return: a dict containing precision, recall and f-score of the function
    """
    if n is None:
        p, r, f = func(reference_summary=reference, created_summary=created, extended_results=True)
    else:
        p, r, f = func(reference_summary=reference, created_summary=created, extended_results=True, n=n)
    return {identifier + recall_str: r, identifier + precision_str: p, identifier + fscore_str: f}


def evaluate(model_id, eval_result_path):
    """
    Coordinates the evaluation of the model after the evaluation texts were created.

    :param model_id: Model to load the evaluation texts.
    :param eval_result_path: path where to find the texts to evaluate
    """
    eval_results = utils.df_from_json(current_path=current_path, path=eval_result_path + model_id)
    eval_results = evaluate_summaries(eval_results)

    utils.create_dir(current_path=current_path, directory_name=eval_result_path, delete=False)
    utils.df_to_json(current_path=current_path, path=eval_result_path + model_id,
                     dataframe=eval_results)


def evaluate_summaries(df):
    """
    Function for evaluation the created summaries. Uses BERTScore and ROUGE-1 -2 -3 -L and prints mean precision,
    recall and fscore

    :param df: Dataframe containing leitsatz_str and summary_str
    :return: the original df with additional columns for precision, recall and fscore
    """
    scores = metrics.bertscore(gold_sum_sents=df[leitsatz_str].values.tolist(),
                               candidate_sum_sents=df[summary_str].values.tolist())
    df_result = df.copy()
    df_result[bertscore_str + precision_str] = scores[0].squeeze()
    df_result[bertscore_str + recall_str] = scores[1].squeeze()
    df_result[bertscore_str + fscore_str] = scores[2].squeeze()
    # rouge-l
    df_result[[rouge_l_str + precision_str, rouge_l_str + recall_str, rouge_l_str + fscore_str]] = \
        df_result.apply(lambda row: apply_rouge(rouge_l, created=row[summary_str],
                                                reference=row[leitsatz_str], identifier=rouge_l_str),
                        axis='columns', result_type='expand')
    # rouge-1
    df_result[[rouge_1_str + precision_str, rouge_1_str + recall_str, rouge_1_str + fscore_str]] = \
        df_result.apply(lambda row: apply_rouge(rouge_n, created=row[summary_str], n=1,
                                                reference=row[leitsatz_str], identifier=rouge_1_str),
                        axis='columns', result_type='expand')
    # rouge-2
    df_result[[rouge_2_str + precision_str, rouge_2_str + recall_str, rouge_2_str + fscore_str]] = \
        df_result.apply(lambda row: apply_rouge(rouge_n, created=row[summary_str], n=2,
                                                reference=row[leitsatz_str], identifier=rouge_2_str),
                        axis='columns', result_type='expand')
    return df_result


def get_tokenizer(model_id):
    """
    Loads the tokenizer for the given model

    :param model_id: Model to load tokenizer for
    :return: the loaded tokenizer
    """
    tokenizer = AutoTokenizer.from_pretrained(models[model_id], trust_remote_code=True)
    tokenizer.padding_side = 'right'
    return tokenizer
