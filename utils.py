import json
import multiprocessing
import os
import re
import shutil

import pandas as pd

import settings

pool_processes = 4
pool_maxtask = 10
pool_chunksize = 30
leitsatz_str = 'leitsatz'
tenor_str = 'tenor'
tatbestand_str = 'tatbestand'
entscheidungsgruende_str = 'entscheidungsgruende'
aktenzeichen_str = 'aktenzeichen'
spruchkoerper_str = 'spruchkoerper'
entscheidungs_datum_str = 'entsch-datum'
gericht_str = 'gericht'
norms_str = 'normen'
rii_text_columns = [leitsatz_str, tenor_str, tatbestand_str, entscheidungsgruende_str]
sentence_marks = ['.', ',', ';', '!', '?']
pp_option_lemmatize = 'preprocessing: lemmatize the text'
pp_option_stopwords = 'preprocessing: remove stopwords'
pp_option_case_normalize = 'preprocessing: normalize cases / put to lower'
pp_option_remove_qout_marks_sing = 'preprocessing: remove qoutation marks around single words'
pp_option_lang_english = 'using english text'
pp_option_lang_german = 'using german text'
no_stopword_list = ['nicht', 'kein']
entsch_gr_start_sentences = ['II.', 'B.', 'B']
rouge1_str = 'rouge1'
rougel_str = 'rougel'
extractive_str = 'extractive'
summary_str = 'summary'
bertscore_str = 'bertscore_'
rouge_l_str = 'rouge_l_'
rouge_1_str = 'rouge_1_'
rouge_2_str = 'rouge_2_'
rouge_3_str = 'rouge_3_'
precision_str = 'precision'
recall_str = 'recall'
fscore_str = 'fscore'
rouge_r_str = 'rouge_recall'
rouge_p_str = 'rouge_precision'
rouge_f_str = 'rouge_f_measure'


def server_path(current_path, path):
    """
    Method to add path in case it is run on server.

    :param current_path: Path to add when run on server
    :param path: Path for local
    :return: Final path for local or server
    """
    if settings.server:
        path = current_path + '/' + path
    return path


def open_file(current_path, path, modes, encoding=None, newline=None):
    """
    Wraps the builtin open function to adjust to server settings

    :param current_path: path of the calling file to adjust for server (without /)
    :param path: Path for file loading relative to calling file
    :param modes: Modes to apply
    :param newline: newline option of the original method, if None nothing will be passed
    :param encoding: encoding option of the original method, if None nothing will be passed
    :return: the opened file
    """
    if encoding is not None:
        return open(server_path(current_path=current_path, path=path), modes, encoding=encoding)
    if newline is not None:
        return open(server_path(current_path=current_path, path=path), modes, newline=newline)
    if newline is not None and encoding is not None:
        return open(server_path(current_path=current_path, path=path), modes, encoding=encoding, newline=newline)
    return open(server_path(current_path=current_path, path=path), modes)


def file_exists(current_path, path):
    """
    Wraps the builtin exists function to adjust to server settings

    :param current_path: path of the calling file to adjust for server (without /)
    :param path: Path for file loading relative to calling file
    :return: True if the file exists
    """
    return os.path.exists(server_path(current_path=current_path, path=path))


def list_dir_files(current_path, path):
    """
    Wraps the builtin os.listdir function to adjust to server settings

    :param current_path: path of the calling file to adjust for server (without /)
    :param path: Path for file loading relative to calling file
    :return: The filenames of the directory
    """
    return os.listdir(server_path(current_path=current_path, path=path))


def df_from_pickle(current_path, path):
    """
    Wraps the pd.read_pickle function to adjust to server settings

    :param current_path: path of the calling file to adjust for server (without /)
    :param path: Path for file loading relative to calling file
    :return: The loaded dataframe
    """
    return pd.read_pickle(server_path(current_path=current_path, path=path))


def df_to_json(current_path, path, dataframe):
    """
    Wraps the df.to_json function to adjust to server settings

    :param current_path: path of the calling file to adjust for server (without /)
    :param path: Path for file loading relative to calling file
    :param dataframe: The dataframe to save
    """
    dataframe.to_json(server_path(current_path=current_path, path=path))


def df_from_json(current_path, path):
    """
    Wraps the json.load function in combination with a dataframe creation to adjust to server settings

    :param current_path: path of the calling file to adjust for server (without /)
    :param path: Path for file loading relative to calling file
    :return: The loaded dataframe
    """
    return pd.DataFrame(json.load(open_file(current_path=current_path, path=path, modes="r")))


def parallel_imap(function, packaged_args):
    """
    Executes the given function in a parallel way. For list data.

    :param function: Function to do in parallel.
    :param packaged_args: Iterable of argumentpairs for each run to be done.
    :return: Result of the parallel work
    """
    if settings.server:
        pool_obj = multiprocessing.Pool(maxtasksperchild=pool_maxtask)
        result = pool_obj.imap(function, packaged_args, chunksize=pool_chunksize)
    else:
        pool_obj = multiprocessing.Pool(processes=pool_processes)
        result = pool_obj.imap(function, packaged_args)
    pool_obj.close()
    pool_obj.join()
    return result


def parallel_apply_async(function, data):
    """
    Executes the given function in a parallel way. For dataframe data.

    :param function: Function to do in parallel.
    :param data: Dataframe to extract the rows for each run to be done.
    :return: Result of the parallel work
    """
    if settings.server:
        pool_obj = multiprocessing.Pool(maxtasksperchild=pool_maxtask)
        result = [pool_obj.apply_async(function, [(index, row)]) for index, row in data.iterrows()]
    else:
        pool_obj = multiprocessing.Pool(processes=pool_processes)
        result = [pool_obj.apply_async(function, [(index, row)]) for index, row in data.iterrows()]
    pool_obj.close()
    pool_obj.join()
    return result


def get_step_subset_raw(steps, path_to_dest_dataframe, source_data, dest_data, call_path):
    """
    Method for stepwise work on datasets. Reads in the already present data and starts
    where last time ended. Used for raw pickle-files in destination

    :param steps: How many rows should be selcted now
    :param path_to_dest_dataframe: Path on where to load the destination data
    :param source_data: Source dataframe to select the rows
    :param dest_data: empty dataframe to load the data into
    :param call_path: path from which the method was called, for server path
    :return: the subset of the source data an the loaded destintion data (source, dest)
    """
    if steps > 0:
        try:
            try:
                var = df_from_pickle(current_path=call_path, path=path_to_dest_dataframe)
            except Exception:
                var = df_from_json(current_path=call_path, path=path_to_dest_dataframe)
            dest_data = pd.concat([dest_data, var], ignore_index=True)
            start = dest_data.shape[0]
        except OSError as _:
            start = 0
        finally:
            end = start + steps
            try:  # case source is a dataframe
                if end >= source_data.shape[0]:
                    return source_data.iloc[start:], dest_data  # subset
                else:
                    return source_data.iloc[start:end], dest_data  # subset
            except Exception:
                if end >= len(source_data):
                    return source_data[start:], dest_data  # subset
                else:
                    return source_data[start:end], dest_data  # subset


def remove_spaces_before_sentence_marks(text):
    """
    Removes unneccessary spaces before '.' etc.

    :param text: Text to replace in
    :return: The cleaned text
    """
    for sentence_mark in sentence_marks:
        while ' ' + sentence_mark in text:
            text = text.replace(' ' + sentence_mark, sentence_mark)
    return text


def remove_brackets(text):
    """
    Removes all matching round bracktet pairs () with their content. Always takes the first brackets that
    appear in the text, so could also be an enumeration like a)

    :param text: Text to remove the brackets from.
    :return: Resulting text
    """
    startindex = text.find('(')
    res = ''
    while startindex > -1:
        endindex = startindex + text[startindex:].find(')')
        if endindex > -1:
            # in case there is a ' ' in front or after the brackets, remove one space
            if startindex > 0 and text[startindex - 1] == ' ':
                startindex -= 1
            # if endindex < len(text) - 1 and text[endindex + 1] == ' ':
            #   endindex += 1
            res += text[:startindex]
            text = text[endindex + 1:]
        else:
            break
        startindex = text.find('(')
    res += text
    return res


def remove_leading_keywords_and_listing_sentences(sentences):
    """
    Method intended for Leitsätze. Some of them start with a single keyword in the first line.
    This is removed. Additionally, Sentences which are only a listin ('1.') will also be removed.

    :param sentences: List of sentences in the original order to remove these things from
    :return: the list of sentences after removing
    """
    # remove leading keywords and sentences which are only enumerations
    sentences_var = list()
    sentence_var = ''
    for i in range(len(sentences)):
        sentence = sentences[i].strip()
        if len(sentence) > 1 and sentence[-1] == '.' and ' ' not in sentence:  # at least two chars
            if any(char.isdigit() for char in sentence) and sentence[0].isdigit():  # most likely an enumeration like '1.'
                continue
        if i > 0 or (i == 0 and len(sentence) > 20):
            # most likely not a short keyword at the beginning
            if sentence[-1] == '.' or sentence[-1] == ',' or sentence[-1] == ':' or \
                    sentence[-1] == ';' or sentence[-1] == '!' or sentence[-1] == '?':
                # sentence end
                sentence_var += sentence
                sentences_var.append(remove_spaces_before_sentence_marks(sentence_var))
                sentence_var = ''
            else:
                # continuing sentence
                sentence_var += sentence + ' '
    return sentences_var


def prepare_leitsatz(l_satz):
    """
    Does the preparation for Leitsätze: First splits into sentences, removes leading keywords and
    single listing sentences and leading listings of sentences

    :param l_satz: Original Leitsatz as one string
    :return: prepared Leitsatz as a list of String
    """
    sentences = split_into_sentences(l_satz)
    sentences = remove_leading_keywords_and_listing_sentences(sentences)
    sentences = [remove_leading_listing(sentence) for sentence in sentences]
    return sentences


def prepare_entsch_gr(raw_string):
    """
    Prepares the entscheidungsgruende. Splits into sentences, removes leading lsitings and selects only part of II.

    :param raw_string: Whole string of the entscheidungsgruende
    :return: list of string with the results
    """
    entschgr = []

    for sentence in split_into_sentences(raw_string):
        first, rest = split_leading_listing(sentence)
        if first is not None:
            entschgr.append(first)
        entschgr.append(rest)

    return select_list_subset(entschgr, entsch_gr_start_sentences)


def select_list_subset(list_of_string, start_strings, end_string=None):
    """
    Selects a subset of a list of strings. If the start_string is not in the list,
    the whole original list is returned. (case-sensitive)
    If more start strings are given, then it will be copied from the first occuring start string.

    sometimes entscheidungsgruende II. is started not with II. but B. Use start_String_2 here

    :param list_of_string: List to get subset from
    :param start_strings: List of Strings to start to copy
    :param end_string: First string where one shouldn't copy anymore, if none is given, then till the end
    :return: Selected subset
    """
    result_list = []
    copy = False
    for i in range(len(list_of_string)):
        string = list_of_string[i]
        if string in start_strings:
            copy = True
        if end_string is not None and string == end_string:
            copy = False
        if copy:
            result_list.append(string)
    # if nothing was found or very little was found
    if len(result_list) == 0 or len(result_list) / len(list_of_string) < 0.2:
        return list_of_string
    return result_list


def abbreviation_ending(text):
    """
    Checks for an input text whether it ends with a known legal abbreviation.
    Known issues: numbers and roman numbering with following dots arent matched

    :param text: Input Text
    :return: True, if it does and with such an abbreviation, False otherwise
    """
    abbrev_list = ['A.', ' a.', 'a.A.', 'a.a.O.', 'ABl.', ' abl.', 'Abs.', ' abs.', 'Abschn.', 'Abse.',
                   ' abzgl.', 'a.D.', 'a.E.', ' a.F.', ' ähnl.', 'a.l.i.c.', ' allg.', ' allgem.',
                   'Alt.', 'AmtsBl.', ' and.', ' angef.', 'Anh.', 'Anl.', 'Anm.', ' Art.', '(Art.', ' aufgeh.',
                   'Aufl.', ' ausf.', 'Ausn.', 'BAnz.', 'BArbBl.', 'BayJMBl.', 'Bd.', 'Bde.', 'Bdg.',
                   'Bearb.', ' begr.', 'Beil.', 'Bek.', ' ber.', ' bes.', 'Beschl.', ' best.', ' bestr.',
                   'Betr.', ' betr.', 'Bf.', 'BGBl.', ' bish.', ' Bl.', 'BPräs.', 'BReg.', 'Bsp.', 'Bst.',
                   'BStBl.', 'BT-Drucks.', 'Buchst.', 'bzgl.', 'bzw.', 'c.i.c.', 'Co.', 'c.p.c.',
                   'c.s.q.n.', 'Ct.', ' dar.', 'Darst.', ' ders.', 'd.h.', 'Diss.', ' div.', 'Dr.',
                   'Drucks.', ' dto.', 'DVBl.', ' ebd.', ' Ed.', 'E.G.', ' eingef.', 'Einf.', 'Einl.',
                   ' einschl.', 'Erg.', ' erk.Sen.', ' erk.', ' Erl.', 'etc.', 'E.U.', ' e.V.',
                   'EVertr.', ' evtl.', 'E.W.G.', ' F.', ' f.', ' Fa.', ' Festschr.', ' ff.', ' Fn.',
                   ' form.', ' fr.', ' fr.Rspr.', ' Fz.', 'GBl.', ' geänd.', 'Gedschr.', ' geg.',
                   ' gem.', 'Ges.', ' gg.', ' ggf.', ' ggü.', ' ggüb.', ' Gl.', ' GMBl.', 'G.o.A.',
                   'Grds.', ' grdsl.', 'Großkomm.', 'Großkomm.z.', 'GVBl.', 'GVOBl.', ' h.A.', 'Halbs.',
                   ' h.c.', 'Hdlg.', 'Hess.', ' heut.', ' heut.Rspr.', ' hins.', ' h.L.', ' h.Lit.',
                   ' h.M.', 'Hrsg.', ' h.Rspr.', 'HS.', 'Hs.', ' i.A.', ' ib.', ' ibd.', ' ibid.',
                   'i.d.', 'i.d.F.', 'i.d.R.', 'i.d.S.', 'i.E.', 'i.e.', 'i.e.S.', 'i.H.d.', 'i.H.v.',
                   'i.K.', ' incl.', ' inkl.', 'inkl.MwSt.', ' insb.', ' insbes.', 'Int.', ' i.O.',
                   ' i.R.', ' i.R.d.', 'i.S.', 'i.S.d.', 'i.S.e.', 'i.S.v.', 'i.ü.', ' iur.', 'i.V.',
                   'i.V.m.', 'i.W.', 'i.Wes.', 'i.w.S.', 'i.Zw.', 'Jahrb.', ' jew.', ' Jh.', 'JMBl.',
                   ' jur.', ' Kap.', ' Ko.', ' krit.', ' kzfr.', 'Lb.', 'Lfg.', 'lgfr.', ' Lief.',
                   'Lit.', ' lit.',  ' lt.', 'Ltd.', 'M.A.', 'm.Änd.', 'MABl.', 'mat.', 'm.a.W.', 'm.E.',
                   ' med.', ' mgl.', 'Mglkt.', 'MinBl.', 'Mio.', ' Mot.', 'M.M.', 'm.N.', 'Mod.',
                   ' mögl.', 'Mot.', 'MünchKomm.', 'm.w.', 'm.w.N.', 'MwSt.', 'Mwst.', 'm.W.v.',
                   'm.zust.Anm.', 'Nachw.', 'Nachw.b.', ' nat.', 'Nds.', 'Neubearb.',  'Neuf.',
                   ' neugef.', 'n.F.', 'Nr.', 'Nrn.', ' o.', 'o.Ä.', ' od.', ' oec.', ' öff.',
                   ' o.g.', ' österr.', 'p.F.V.', ' pharm.', ' phil.', ' pol.', 'Postf.', ' pp.',
                   ' ppA.', ' ppa.', 'Prof.', 'Prot.', ' publ.', ' p.V.', 'p.V.V.', 'q.e.d.',
                   'RdErl.', 'Rdn.', 'Rdnr.', 'RdSchr.', ' rel.', ' rer.', 'RGBl.', 'Rn.', 'Rspr.',
                   'Rz.', 'S.', ' s.', 's.a.', 'Schr.', ' scil.', 'Sen.', ' sinngem.', 'SiZess.',
                   'Slg.', 's.o.', ' sog.', 'Sonderbeil.', 'Stpfl.', ' str.', ' st.', 'st.Rspr.',
                   ' st. Rspr.', 'stud.iur.', 's.u.', ' teilw.', ' theol.', 'Thür.', ' TO.', ' tw.',
                   'Tz.', ' u.', 'u.a.', 'UAbs.', 'u.a.m.', ' umstr.', ' unmgl.', 'Unmglkt.', ' unmögl.',
                   'Urt.', ' usw.', ' u.U.', ' V.', ' v.', 'Var.', 'Ver.', ' vgl.', 'V.m.', 'VOBl.',
                   'Vor.', 'Vorbem.', 'Warn.', ' weg.', ' wg.', 'W.G.G.', 'w.z.b.w.', 'z.B.', 'z.Hd.',
                   'Ziff.', 'z.T.', ' zust.', 'zust.Anm.', ' zw.' 'z.Z.', ' zzgl.', ';',
                   'II.1.a.',  '(s.',
                   ]
    for abbrev in abbrev_list:
        if text.endswith(abbrev):
            return True
    if len(text) >= 3 and re.search(" .\\.", text[-3:]):
        return True
    return False


def remove_leading_listing(sentence):
    """
    removes leading listings / enumerations like 1. or a)

    :param sentence: Sentence to remove from
    :return: Processed sentence
    """
    return split_leading_listing(sentence)[1]


def split_leading_listing(sentence):
    """
    Splits the sentence from a possible listing (1. or a) ) at the start.

    :param sentence: Sentence to split
    :return: (start, rest) with start being the listing or None, if there is no listing and
                rest being the rest of the sentence or the original sentence if there was no listing
    """
    first_word = sentence.split()
    if first_word is None or len(first_word) == 0:
        first_word = ''
    else:
        first_word = first_word[0]
    rest = sentence[len(first_word) + 1:]
    # could be a name like M. Leicht
    if (first_word.endswith('.') or first_word.endswith(')')) and len(rest.split()) > 1 and first_word != 'Art.':
        # Enumeration!
        return first_word, rest
    else:
        return None, sentence


def split_into_sentences(input_text, language=pp_option_lang_german):
    """
    Splits text into sentences. Uses spacy sentences but fixes broken sentences on \n or Abbreviations

    :param input_text: Text to split into sentences
    :return: A list of sentences which where split
    """

    paragraphs = input_text.split('\n')
    sentences = list()
    sentence_var = ''
    # roughly split original leitsatz into sentences
    for paragraph in paragraphs:
        if language==pp_option_lang_english:
            nlp_paragraph = settings.nlp_english(paragraph)
        else:
            nlp_paragraph = settings.nlp(paragraph)
        sentences_paragraph = []
        for sent in nlp_paragraph.sents:
            sent = sent.text.strip()
            # some leading listings aren't detected
            a, b = split_leading_listing(sent)
            if a is not None:
                sentences_paragraph.append(a)
            sentences_paragraph.append(b)
        for i in range(0, len(sentences_paragraph)):
            # add a space before next token if it isn't a sentence mark
            if not (sentences_paragraph[i].startswith('.') or sentences_paragraph[i].startswith(':')
                    or sentences_paragraph[i].startswith('?') or sentences_paragraph[i].startswith('!')):
                sentence_var += ' '
            sentence_var += sentences_paragraph[i]
            # if not sentence_var.count('(') > sentence_var.count(
            #        ')') and not sentence_var.strip() == '':  # no unclosed brackets
            if (sentences_paragraph[i].endswith('.') or sentences_paragraph[i].endswith(':')
                or sentences_paragraph[i].endswith('?') or sentences_paragraph[i].endswith('!')) \
                    and not abbreviation_ending(sentence_var) \
                    and not sentence_var.strip() == '':
                # Satz sehr wahrscheinlich wirklich zuende
                sentences.append(sentence_var.strip())
                sentence_var = ''
        if not sentence_var.strip() == '':
            #        if not sentence_var.count('(') > sentence_var.count(
            #               ')') and not sentence_var.strip() == '':  # no unclosed brackets
            sentences.append(sentence_var.strip())  # am Ende des Paragraphen soll auch fertig sein
            sentence_var = ''
    # end of whole text
    if sentence_var.strip() != '':
        sentences.append(sentence_var.strip())
    return sentences


def preprocess_text(text, options):
    """
    Allows simple preprocessing like lemmatization on strings.

    :param text: Text to preprocess
    :param options: Options specifying on what preprocessing is to be done, if None, text will be returned
    :return: the preprocessed text, if text is None, the result will also be ''
    """
    if text is None:
        return ''
    if options is None:
        return text
    if pp_option_lang_english in options:
        text_spacy = settings.nlp_english(text)
    else:
        text_spacy = settings.nlp(text)
    result_text = ''
    for token in text_spacy:
        # stop-words removing: no stopwords or stopwords shouldn't be removed
        if not token.is_stop or pp_option_stopwords not in options or token.lemma_ in no_stopword_list:
            # lemmatization if wanted
            if pp_option_lemmatize in options and token.text not in sentence_marks:
                to_append = token.lemma_
            else:
                to_append = token.text
            if pp_option_remove_qout_marks_sing in options and to_append[0] == '"' and to_append[-1] == '"':
                to_append = to_append.replace('"', '')
            result_text += to_append + ' '
    result_text = result_text.strip()
    # case-normlaization, all to lower
    if pp_option_case_normalize in options:
        return result_text.lower()
    else:
        return result_text


def create_dir(current_path, directory_name, delete=True):
    """
    Creates a directory if it doesn't exist

    :param current_path: path of the calling file
    :param directory_name: name / path to create
    :param delete: if True, than an old directory with same name will be delted
    """
    if delete and file_exists(current_path=current_path, path=directory_name):
        shutil.rmtree(server_path(current_path=current_path, path=directory_name))
    if not file_exists(current_path=current_path, path=directory_name):
        os.makedirs(server_path(current_path=current_path, path=directory_name))


def filter_topical_leitsaetze(data):
    """
    Filters the guiding principles to exclude topical ones (without Zur, Zum....)

    :param data: a dataframe with column 'leitsatz'
    :return: a dataframe withot topical guiding principles
    """
    data = data[data[leitsatz_str].apply(lambda x: not x.startswith('Zur '))]
    data = data[data[leitsatz_str].apply(lambda x: not x.startswith('Zum '))]
    data = data[data[leitsatz_str].apply(lambda x: not x.startswith('Zu de'))]
    return data

