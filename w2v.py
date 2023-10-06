import jieba
import pandas as pd
from gensim.models import Word2Vec
import matplotlib.pyplot as plt
from gensim.models import KeyedVectors
import csv
import numpy as np
import os

csvs, stop_words, dicts = [], [], []


def load_stopwords(stopwords_path):
    stopwords = []
    with open(stopwords_path, "r", encoding="utf-8") as f:
        for line in f.readlines():
            for word in line.split():
                stopwords.append(str(word))
    return stopwords

def read_file(file_name):
    df = pd.read_csv(file_name, encoding="utf-8")
    return df["comment"]

def get_dict(file_name):
    Dict = []
    with open(file_name, "r", encoding="utf-8") as f:
        for line in f.readlines():
            for word in line.split():
                Dict.append(str(word))
    for word in Dict:
        jieba.add_word(word)
    return Dict


def use_stopwords_dic(brand):
    stopwords_ = "stopwords.txt"
    dict_ = "dict.txt"
    for item in stop_words:
        if brand in item:
            stopwords_ = item
    for item in dicts:
        if brand in item:
            dict_ = item

    return stopwords_, dict_


def preprosses(data, stop_words):
    processed_data = []
    processed_dict = []
    for line in data:
        line = str(line)
        words = jieba.cut(line)
        words = [word for word in words if word not in stop_words]
        processed_data.append(words)

    for comment in processed_data:
        mergedcomment = comment
        processed_dict.append(mergedcomment)
    
    return processed_data, processed_dict


def w2v(text,mincount):
    model = Word2Vec(text,min_count=mincount)
    model.save(f"{output_path}/{csvs[num].split('.')[0]}/w2v.model")

def w2v2xlsx(_freq):
    model = KeyedVectors.load(f"{output_path}/{csvs[num].split('.')[0]}/w2v.model")
    word_vectors = model.wv
    highest_freq = word_vectors.index_to_key[:_freq]
    vectors = [word_vectors[word] for word in highest_freq]
    similarity_matrix = []
    for i in range(len(vectors)):
        similarity_matrix.append([])
        for j in range(len(vectors)):
            similarity_matrix[i].append(word_vectors.similarity(highest_freq[i], highest_freq[j]))


    df = pd.DataFrame(similarity_matrix, index=highest_freq, columns=highest_freq)
    df.to_excel(f"{output_path}/{csvs[num].split('.')[0]}/w2v.xlsx")

    
if __name__ == "__main__":
    current_path = os.getcwd()
    csvs_dir=os.path.join(current_path, 'csvs/')
    dicts_dir=os.path.join(current_path, 'dicts/')
    stop_words_dir=os.path.join(current_path, 'stopwords/')
    output_path=os.path.join(current_path, 'result/')
    csvs = []
    for root, dirs, files in os.walk(csvs_dir):
        for file in files:
            if os.path.splitext(file)[1] == '.csv':
                csvs.append(file)

    dicts = []
    for root, dirs, files in os.walk(dicts_dir):
        for file in files:
            if os.path.splitext(file)[1] == '.txt':
                dicts.append(file)

    stop_words = []
    for root, dirs, files in os.walk(stop_words_dir):
        for file in files:
            if os.path.splitext(file)[1] == '.txt':
                stop_words.append(file)

    print(csvs)
    print(dicts)
    print(stop_words)
    for num in range(len(csvs)):
        print (f"正在处理{csvs[num]}")
        brand = csvs[num].split("_")[0]
        stopwords_, dict_ = use_stopwords_dic(brand)
        stop_word = load_stopwords(stop_words_dir + stopwords_)
        Dict = get_dict(dicts_dir + dict_)
        file = read_file(csvs_dir+csvs[num])
        processed_data, processed_dict = preprosses(file, stop_word)
        w2v(processed_dict, 5)
        w2v2xlsx(100)
        for item in Dict:
            jieba.del_word(item)
        print (f"{csvs[num]}处理完成")


    