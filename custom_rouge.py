# implementing rouge scores as there was now german implementation
import utils

beta = 1
# https://aclanthology.org/W04-1013/
rouge_standard_pp_options = [utils.pp_option_stopwords]


def rouge_n(reference_summary, created_summary, n, pp_options=rouge_standard_pp_options,
            extended_results=False, language=None):
    """
    Calculates the rouge n score

    :param reference_summary: gold standard summary
    :param created_summary: summary to evaluate
    :param n: size of n-grams
    :param pp_options: list of options for preprocessing, if None then no preprocessing will be done
    :param extended_results: indicates, whether, precision, recall and f-measure should be returned
    :param language: for specifying english languge
    :return: the score or (precision, recall, f-measure) if extended results are wanted
    """
    if language is not None:
        pp_options = pp_options + [language]
    # preprocess
    if pp_options is not None:  # otherwise don't preprocess. Text is already preprocessed
        reference_summary = utils.preprocess_text(reference_summary, pp_options)
        created_summary = utils.preprocess_text(created_summary, pp_options)
    else:  # seperate sentence marks from tokens
        for sentence_mark in utils.sentence_marks:
            reference_summary = reference_summary.replace(sentence_mark, ' '+sentence_mark)
            created_summary = created_summary.replace(sentence_mark, ' ' + sentence_mark)
    # split into n-grams of size n
    # count occurances of single ngrams
    reference_ngrams, ref_complete_count = count_n_grams(reference_summary, n)
    created_ngrams, created_complete_count = count_n_grams(created_summary, n)

    overlapping_count = 0
    for ref_key in reference_ngrams.keys():
        created_count = created_ngrams.get(ref_key)
        if created_count is not None:  # ngrams in both dicts
            ref_count = reference_ngrams[ref_key]
            overlapping_count += min(ref_count, created_count)

    # calculate score
    if ref_complete_count == 0 or created_complete_count == 0:
        if extended_results:
            return 0,0,0
        return 0
    recall = overlapping_count / ref_complete_count
    if extended_results:
        precision = overlapping_count / created_complete_count
        fscore = 0  # setting to 0, maybe another value is better?
        if precision+recall > 0:
            fscore = (2 * precision * recall) / (precision + recall)
        return precision, recall, fscore
    return recall


def count_n_grams(pp_summary, n):
    """
    Counts the n-grams of the given size in a summary.

    :param pp_summary: Pre-processed summary
    :param n: n for the size of ngrams
    :return: {ngram:count} for all ngrams in the summary
    """
    words = pp_summary.split(' ')
    complete_count = 0
    n_grams = {}
    for i in range(len(words)-(n-1)):
        n_gram = ' '.join(words[i:i+n])
        if n_gram != '':
            complete_count += 1
            count = n_grams.get(n_gram)
            if count is None:
                count = 0
            n_grams[n_gram] = count + 1
    return n_grams, complete_count


def rouge_l(reference_summary, created_summary, pp_options=rouge_standard_pp_options, extended_results=False, language=None):
    """
    Calculates the rouge-l value of a summary and its gold standard summary

    :param reference_summary: Gold standard summary
    :param created_summary: Created summary to compare
    :param pp_options: options for preprocessing, if None then there will be no preprocessing
    :param extended_results: if True, precision, recall and f-score will be returned
    :param language: for specifying english languge
    :return: The calculated score, if extended results are wanted (precision, recall, f-measure)
    """
    if language is not None:
        pp_options = pp_options + [language]
    # preprocess
    if pp_options is not None:  # otherwise don't preprocess. Text is already preprocessed
        reference_summary = utils.preprocess_text(reference_summary, pp_options)
        created_summary = utils.preprocess_text(created_summary, pp_options)
    # seperate sentence marks from words
    # split into sentences
    m_reference_word_number = len(reference_summary.split(' '))
    reference_summary = utils.split_into_sentences(reference_summary)
    n_created_word_number = len(created_summary.split(' '))
    created_summary = utils.split_into_sentences(created_summary)
    total_sum_subsequences = 0
    # to make sure every word in the created summary is used only once
    used_created_indices = [set()]*len(created_summary)
    used_gold_indices = [set()]*len(reference_summary)
    for j in range(len(reference_summary)):
        ref_sentence = reference_summary[j]
        # calculate union longest subsequence
        for i in range(len(created_summary)):
            created_sentence = created_summary[i]
            indices_a, indices_b = get_subsequence(ref_sentence, created_sentence)
            used_gold_indices[j] = (used_gold_indices[j]).union(indices_a)
            used_created_indices[i] = (used_created_indices[i]).union(indices_b)
    # used indices of b here to ensure words arent used twice
    used_created_indices = [len(sent_set) for sent_set in used_created_indices]
    used_gold_indices = [len(sent_set) for sent_set in used_gold_indices]
    total_sum_subsequences = min(sum(used_gold_indices), sum(used_created_indices))

    if total_sum_subsequences == 0:
        if extended_results:
            return 0,0,0
        return 0
    p_lcs = total_sum_subsequences / n_created_word_number
    r_lcs = total_sum_subsequences / m_reference_word_number
    f_lcs = ((1 + beta * beta) * r_lcs*p_lcs) / (r_lcs + beta * beta * p_lcs)
    if extended_results:
        return p_lcs, r_lcs, f_lcs
    return f_lcs


def get_subsequence(sent_a, sent_b):
    """
    Finds all (not necessarily consecutive) subsequences of a in b.
    :param sent_a: Sentence to find subsequences from
    :param sent_b: Sentence to find subsequence in
    :return: (ind_a, ind_b) two sets of indices of sent_a and sent_b of the longest subsequence
    """
    result_a = set()
    words_a = sent_a.split(' ')
    words_b = sent_b.split(' ')
    for word_index_a in range(len(words_a)):
        word_result = set()
        char_index_b = 0
        while word_index_a < len(words_a):
            # word is contained
            try:
                found_index = words_b.index(words_a[word_index_a], char_index_b)
                word_result.add(word_index_a)
                char_index_b = found_index
                word_index_a += 1
            except ValueError:
                # word not in b contained, do nth
                word_index_a += 1
        if len(word_result) > len(result_a):
            result_a = word_result
    result_b = set([words_b.index(words_a[a_ind]) for a_ind in result_a])
    return result_a, result_b


if __name__ == "__main__":

    print('Done')
