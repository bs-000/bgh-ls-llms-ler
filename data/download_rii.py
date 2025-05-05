import time
import xml.etree.ElementTree as ET
import pprint
import urllib.request as request
import zipfile
import os

import pandas as pd

import settings
import utils
from utils import time_convert

base_dir_bgh = 'raw_data/BGH_Data'
extended_dir_bgh = base_dir_bgh + '/senates'
dataframe_dir_bgh = 'dataframes/bgh/'
base_dir_bverwg = 'raw_data/BVerwG_Data'
extended_dir_bverwg = base_dir_bverwg + '/senates'
pickle_name_bgh = 'bgh_data.pkl'
pickle_name_bgh_lemmatized = 'bgh_data_lemmatized.pkl'
pickle_name_bgh_lemmatized_sw = 'bgh_data_lemmatized_no_stopwords.pkl'
pickle_name_bverwg = 'bverwg_data.pkl'
simple_attributes = ["doknr", "ecli", "gertyp", "gerort", "spruchkoerper", "entsch-datum",
                     "aktenzeichen", "doktyp", "norm", "vorinstanz", "mitwirkung", "titelzeile",
                     "leitsatz", "sonstosatz", "tenor", "tatbestand", "entscheidungsgruende",
                     "gruende", "abwmeinung", "sonstlt", "identifier", "coverage", "language",
                     "publisher", "accessRights"]
nested_attributes = ["region_abk", "region_long"]
text_attributes = ["titelzeile", "leitsatz", "sonstosatz", "tenor", "tatbestand",
                   "entscheidungsgruende", "gruende", "abwmeinung", "sonstlt"]
stopword_extension = '_no_stopwords'
current_path = 'data'


def get_file_list():
    """
    Makes http request for the files
    :return: the web page with all current cases as an xml-tree
    """
    xml_file, https_message = request.urlretrieve('https://www.rechtsprechung-im-internet.de/rii-toc.xml')
    tree = ET.parse(xml_file)
    root = tree.getroot()
    return root


def count_senat_cases(base_dir, tag):
    """
    counts the cases per senate and writes them to the file senat_counts.txt
    :param base_dir: Directory for saving
    :param tag: tag to recognize the court
    """
    root = get_file_list()
    folders = {}
    for child in root:
        text = child[0].text
        if tag in text:
            folders[text] = folders.get(text, 0) + 1
    with utils.open_file(current_path=current_path, path=base_dir + '/senat_counts.txt', modes='w') as f:
        pprint.pprint(folders, stream=f)


def count_cases(root, tag):
    """
    counts all cases belonging to the given tag and returns the count
    :param root: downloaded xml-tree with all files
    :param tag: tag to find in the name
    :return: number of cases belonging to the BGH
    """
    count = 0
    for child in root:
        if tag in child[0].text:
            count += 1
    return count


def download(base_dir, extended_dir, tag):
    """
    download all cases to a folder related to their senats
    :param base_dir: Name of the directory for the data
    :param extended_dir: name of the subdirectory for saving
    :param tag: tag to recognize the court (BGH, BVerwG)
    """
    # set up directories
    utils.create_dir(current_path=current_path, directory_name=base_dir)
    utils.create_dir(current_path=current_path, directory_name=extended_dir)
    # do the download
    root = get_file_list()  # 0 ist gericht, 3 ist link
    max_cases = count_cases(root, tag)
    downloaded = 0
    for child in root:
        while True:
            try:
                if tag in child[0].text:
                    filename, http = request.urlretrieve(child[3].text)
                    with zipfile.ZipFile(filename, 'r') as zip_ref:
                        zip_ref.extractall(
                            utils.server_path(current_path=current_path,
                                              path=extended_dir + '/' + child[0].text.replace('\n', '') + '/'))
                    os.remove(filename)
                    downloaded += 1
                    print("\rDownloaded %d of %d " % (downloaded, max_cases) + tag + "Cases", end="")
            finally:
                break
    print("\nDone!")


def read_file_data(file):
    """
    Reads the data of one case / file.

    :param file: package containing (filename, directory, directory extension) to address the file
    :return: a dictionary with key: attribute_name and value: attribute_value
    """
    filename, directory, extended_dir = file
    tree = ET.parse(utils.server_path(current_path=current_path, path=os.path.join(extended_dir, directory, filename)))
    root = tree.getroot()
    res = {}
    for attribute in simple_attributes:
        attr = root.find(attribute)  # leitsatz überprüfen: zwei Worte zusammen, aber leerzeichen immer noch da!
        text = ''
        for t in attr.itertext():
            if t == '.' or t == ',' or t == ';' or t == '!' or t == '?':
                text = text.strip()     # remove space before these characters
            text += t + ' '
        text = text.strip()
        if text == '':
            res[attribute] = None
        else:
            res[attribute] = text

    for attribute in nested_attributes:
        nesting = attribute.split('_')
        xml_tag = root
        # find nested attribute
        for i in range(len(nesting)):
            xml_tag = xml_tag.find(nesting[i])
        text = ""
        for t in xml_tag.itertext():
            if t == '.' or t == ',' or t == ';' or t == '!' or t == '?':
                text = text.strip()     # remove space before these characters
            text += t + ' '
        text = text.strip()
        if text == '':
            res[attribute] = None
        else:
            res[attribute] = text

    for attribute in utils.rii_text_columns:
        if res[attribute] is not None:
            if settings.remove_brackets:
                res[attribute] = utils.remove_brackets(res[attribute])
            res[attribute] = utils.remove_spaces_before_sentence_marks(res[attribute])

    return pd.DataFrame(res, index=[0])


def create_pickle(extended_dir, pickle_name, steps):
    """
    Combines all downloaded files of the given extended directory into one pickle

    :param extended_dir: extended dir to find the files
    :param pickle_name: name of the pickle to save
    :param steps: how many cases should be worked on now
    """
    utils.create_dir(current_path=current_path, directory_name=dataframe_dir_bgh, delete=False)
    start_time = time.time()
    extension = ''
    if settings.remove_brackets:
        extension = settings.no_brackets_suffix

    files = [(filename, directory, extended_dir) for directory in
             utils.list_dir_files(current_path=current_path, path=extended_dir) for filename in
             utils.list_dir_files(current_path=current_path, path=os.path.join(extended_dir, directory))
             if filename.endswith(".xml")]

    original_length = len(files)
    data = pd.DataFrame(columns=simple_attributes + nested_attributes)

    pickle_path = dataframe_dir_bgh+extension+pickle_name

    files, data = utils.get_step_subset_raw(steps=steps,
                                            path_to_dest_dataframe=pickle_path,
                                            source_data=files,
                                            dest_data=data,
                                            call_path=current_path)

    result = utils.parallel_imap(read_file_data, files)
    for row in result:
        data = pd.concat([data, row], ignore_index=True)
    with utils.open_file(current_path=current_path, path=pickle_path, modes='wb') as f:
        data.to_pickle(f)

    print('Resulting dataframes have length ' + str(data.shape[0]) +
          ' (' + str(data.shape[0] / original_length * 100) + '%)')
    end_time = time.time()
    time_lapsed = end_time - start_time
    time_convert(time_lapsed)


def get_selected_bgh_data(case, directory='.\\'):
    """
    Shortcut for getting the BGH data currently needed. Selects all data from the Civil copurts which contain 'Urteile'

    :param case: What data to retrieve: 0 - original, 1 - lemmatized, 2 - lemmatized without stopwords, else None
    :param directory: directory offset from current position, with ending slashes
    :return: the data
    """
    if case == 0:
        return get_data(pickle_name_bgh, directory, spruchkoerper='Zivilsenat', doktyp='Urteil')
    if case == 1:
        return get_data(pickle_name_bgh_lemmatized, directory, spruchkoerper='Zivilsenat', doktyp='Urteil')
    if case == 2:
        return get_data(pickle_name_bgh_lemmatized_sw, directory, spruchkoerper='Zivilsenat', doktyp='Urteil')
    return None


def lemmatize_row(package):
    """
    Takes a dataframe row and lematizes the text attributes defined in text_attributes. It adds entries for simple
    lemmatization and the removal of stopwords

    :param package: (index,row) as a series
    :return: series with the lemmatized data
    """
    _, row = package
    res = {}
    for key in simple_attributes + nested_attributes:
        value = row.get(key)
        if key in text_attributes and value is not None:
            text_data = value
            doc = settings.nlp(text_data)
            result_text = ''
            result_text_no_stopwords = ''
            for token in doc:
                result_text += token.lemma_ + ' '
                if not token.is_stop:
                    result_text_no_stopwords += token.lemma_ + ' '
            res[key] = result_text.strip()
            res[key + stopword_extension] = result_text_no_stopwords.strip()
        else:
            res[key] = value
    return pd.Series(res)


def create_lemmatized_pickles(steps=0):
    """
    Creates the lemmatized pickles from the original data pickle - if that pickle does not exist, your get
    an error. Lastly prints the current dataframe sizes and percentages (for stepwise creation).

    :param steps: How much data from start should be processed. The start is calculated from the length of the
    currently saved dataframes.
    """
    start_time = time.time()
    extension = ''
    if settings.remove_brackets:
        extension = settings.no_brackets_suffix

    pickle_path_lem = dataframe_dir_bgh + extension + pickle_name_bgh_lemmatized
    pickle_path_lem_no_sw = dataframe_dir_bgh + extension + pickle_name_bgh_lemmatized_sw
    original_data = get_data(pickle_name_bgh)

    original_length = original_data.shape[0]
    data_lemmatized = pd.DataFrame(columns=simple_attributes + nested_attributes)
    _, data_lemmatized = utils.get_step_subset_raw(steps=steps,
                                                   path_to_dest_dataframe=pickle_path_lem,
                                                   source_data=original_data,
                                                   dest_data=data_lemmatized,
                                                   call_path=current_path)

    data_no_stopwords = pd.DataFrame(columns=simple_attributes + nested_attributes)
    original_data, data_no_stopwords = utils.get_step_subset_raw(steps=steps,
                                                                 path_to_dest_dataframe=pickle_path_lem_no_sw,
                                                                 source_data=original_data,
                                                                 dest_data=data_no_stopwords,
                                                                 call_path=current_path)

    result = utils.parallel_apply_async(lemmatize_row, original_data)

    for row in result:
        combined_row = row.get()
        lemmatized = {}
        no_stopwords = {}
        for key in combined_row.index:
            if key in simple_attributes + nested_attributes:
                lemmatized[key] = combined_row[key]
                if key not in text_attributes:
                    no_stopwords[key] = combined_row[key]
            else:  # no_stopword attributes
                real_key = key[:-len(stopword_extension)]
                no_stopwords[real_key] = combined_row[key]

        data_lemmatized = pd.concat([data_lemmatized, pd.DataFrame(pd.Series(lemmatized)).T], axis=0, ignore_index=True)
        data_no_stopwords = pd.concat([data_no_stopwords, pd.DataFrame(pd.Series(no_stopwords)).T], ignore_index=True)
    with utils.open_file(current_path=current_path, path=pickle_path_lem, modes='wb') as f:
        data_lemmatized.to_pickle(f)
    with utils.open_file(current_path=current_path, path=pickle_path_lem_no_sw, modes='wb') as f:
        data_no_stopwords.to_pickle(f)
    print('Resulting dataframes have length ' + str(data_no_stopwords.shape[0]) +
          ' (' + str(data_no_stopwords.shape[0] / original_length * 100) + '%)')
    end_time = time.time()
    time_lapsed = end_time - start_time
    time_convert(time_lapsed)


def get_data(pickle_name, directory='../data/', spruchkoerper=None, doktyp=None):
    """
    Method for access to the bgh pickle
    :param pickle_name: name to identify the data
    :param directory: directory path to the data file (with ending slash)
    :param spruchkoerper: Parameter can be used to select the senates (checks whether the given string is contained
    in the datas spruchkoerper)
    :param doktyp: can be used to select specific documents (like 'Urteil', 'Beschluss', etc.), must contain the word
    :return: The data as a pandas dataframe
    """
    extension = ''
    if settings.remove_brackets:
        extension = settings.no_brackets_suffix
    data = utils.df_from_pickle(current_path=current_path, path=directory + dataframe_dir_bgh + extension + pickle_name)
    if spruchkoerper is not None:
        data = data[data['spruchkoerper'].notnull()]
        data = data[data['spruchkoerper'].str.contains(spruchkoerper)]
    if doktyp is not None:
        data = data[data['doktyp'].str.lower().str.contains(doktyp.lower())]
    data = data.dropna(axis=1, how='all')  # drop all columns with no value
    data = data.drop_duplicates()
    return data


if __name__ == "__main__":
    # last download BGH: 07.07.2022, 21868 cases
    # last download BGH raw_data: 19.10.2022, 22342 cases
    # download(base_dir=base_dir_bgh, extended_dir=extended_dir_bgh, tag='BGH')
    # count_senat_cases(base_dir=base_dir_bgh, tag='BGH')

    create_pickle(extended_dir=extended_dir_bgh, pickle_name=pickle_name_bgh, steps=2)
    create_lemmatized_pickles(steps=1) # 3000 dauert 8-9h
    v = get_selected_bgh_data(0)

    print('Done with main')
