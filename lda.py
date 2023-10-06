import pandas as pd
import jieba
import os
import gensim
from gensim import corpora, models
import matplotlib.pyplot as plt
import pyLDAvis.gensim

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
        words = [word for word in words if ((word not in stop_words) and (len(word) > 1))]
        processed_data.append(words)

    for comment in processed_data:
        mergedcomment = comment
        processed_dict.append(mergedcomment)
    
    return processed_data, processed_dict

def lda_pre(Dict):
    dictionary = corpora.Dictionary(Dict)
    corpus = [dictionary.doc2bow(text) for text in Dict]
    # 创建文件夹并保存字典和语料库
    if not os.path.exists(f"{output_path}/{csvs[num].split('.')[0]}/"):
        os.makedirs(f"{output_path}/{csvs[num].split('.')[0]}/")
    corpora.MmCorpus.serialize(f"{output_path}/{csvs[num].split('.')[0]}/.mm", corpus)
    # frequency
    xdi={ v:k for k,v in dictionary.token2id.items()}
    fre=dictionary.cfs
    li = sorted(fre.items(), key=lambda d:d[1], reverse = True)
    l=[(xdi[i[0]],i[1]) for i in li]
    df = pd.DataFrame(l, columns=['word','fre'])
    df.to_csv(f"{output_path}/{csvs[num].split('.')[0]}/fre.csv", index=False)
    return dictionary, corpus

def eval_lda(processed_dict,start_topic=2,end_topic=5):
    start_topic = start_topic
    end_topic = end_topic
    best_coherence = -1
    best_topic = -1
    scores=[]
    dictionary, corpus = lda_pre(processed_dict)
    for i in range(start_topic, end_topic):
        lda_model = gensim.models.ldamodel.LdaModel(corpus=corpus, id2word=dictionary, num_topics=i, passes=10)
        lda_model.save(f"{output_path}/{csvs[num].split('.')[0]}/.model")
        print(lda_model.show_topics(num_topics=i,num_words=10, log=False, formatted=True))

        vis = pyLDAvis.gensim.prepare(lda_model, corpus, dictionary)
        pyLDAvis.save_html(vis, f"{output_path}/{csvs[num].split('.')[0]}/" + f'{i}.html')

        coherence_model_lda = gensim.models.CoherenceModel(model=lda_model, texts=processed_dict, dictionary=dictionary, coherence='c_v')
        coherence_lda = coherence_model_lda.get_coherence()
        scores.append(coherence_lda)
        print(f"Topic {i}: {coherence_lda}")
        if coherence_lda > best_coherence:
            best_coherence = coherence_lda
            best_topic = i

    # 画图
    plt.plot(range(start_topic, end_topic), scores)
    plt.xlabel("Number of Topics")
    plt.ylabel("Coherence Score")
    plt.savefig(f"{output_path}/{csvs[num].split('.')[0]}/" + 'lda.png')
    plt.show()
    
    # save figure

    

    df = pd.DataFrame(scores, columns=['score'])
    df.to_csv(f"{output_path}/{csvs[num].split('.')[0]}/score.csv", index=False)
    return best_topic


def load_lda(path):
    dictionary = corpora.Dictionary.load(path + '.model.id2word')
    lda = models.LdaModel.load(path + '.model')
    corpus = corpora.MmCorpus(path + '.mm') 
    vis = pyLDAvis.gensim.prepare(lda, corpus, dictionary)
    pyLDAvis.save_html(vis, path + '.html')
    

if __name__ == "__main__":
    current_path = os.getcwd()
    csvs_dir=os.path.join(current_path, 'csvs/')
    dicts_dir=os.path.join(current_path, 'dicts/')
    stop_words_dir=os.path.join(current_path, 'stopwords/')
    output_path=os.path.join(current_path, 'result/')
    # 遍历csvs_dir下的所有csv文件，将文件名存入csvs列表中
    csvs = []
    for root, dirs, files in os.walk(csvs_dir):
        for file in files:
            if os.path.splitext(file)[1] == '.csv':
                csvs.append(file)
    # 遍历dicts_dir下的所有txt文件，将文件名存入dicts列表中
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
        
        stop_word_path, Dict_word_path = use_stopwords_dic(csvs[num].split('.')[0].split('_')[0])
        stop_word = load_stopwords(stop_words_dir+stop_word_path)
        file = read_file(csvs_dir+csvs[num])
        Dict = get_dict(dicts_dir+Dict_word_path)
        print (f"stopwords={stop_word_path}")
        print (f"dicts={Dict_word_path}")
        processed_data, processed_dict = preprosses(file, stop_word)
        dictionary = corpora.Dictionary(processed_dict)
        num_topics = eval_lda(processed_dict,2,11) #评估可以加参数，默认从2-10
        for item in Dict:
            jieba.del_word(item)
        print(f"最优主题数为{num_topics}")


        # lda_model.save(f"{output_path}/{csvs[num].split('.')[0]}/.model")
        # load_lda(f"{output_path}/{csvs[num].split('.')[0]}/")
