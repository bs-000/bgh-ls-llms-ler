
import torch.cuda
from bert_score import BERTScorer

device = 'cpu'
if torch.cuda.is_available():
    device = 'cuda'
scorer = BERTScorer(model_type='microsoft/mdeberta-v3-base', device=device)  # lang="de",


bert_score = 'Bert Score'


def bertscore(gold_sum_sents, candidate_sum_sents):
    """
    Calculates BERTScore, https://github.com/Tiiiger/bert_score/blob/master/bert_score/scorer.py
    If lists are given, then index 0 of gold summary belongs to index 0 of candidate summaries etc.

    :param gold_sum_sents: String of gold summary sentences or list of gold_sumaries
    :param candidate_sum_sents: String of candidate summary sentences or list of candidate summaries
    :return: precision, recall, fscore (list, if input is list)
    """
    results = scorer.score(cands=candidate_sum_sents, refs=gold_sum_sents, batch_size=1)
    return results
