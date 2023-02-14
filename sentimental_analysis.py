import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import re

from konlpy.tag import Okt
from collections import Counter
from wordcloud import WordCloud

import warnings

warnings.filterwarnings("ignore")



# result 변수 반환
# 사용법 : apply_regular_expression(df['text'][숫자])
def apply_regular_expression(text):
    import re
    hangul = re.compile('[^ ㄱ-ㅣ가-힣]')
    result = hangul.sub('', text)
    return result


# nouns 변수 반환
# 사용법 : returnNouns(df) <-- ['text']는 쓰지말것.
def returnNouns(dataframe):
    from konlpy.tag import Okt
    nouns_tagger = Okt()
    nouns = nouns_tagger.nouns(apply_regular_expression("".join(dataframe['text'].tolist())))
    return nouns


# nouns_tagger 변수 반환
# 이거 뒤에가서 씀.
def returnNouns_tagger():
    nouns_tagger = Okt()
    return nouns_tagger


# counter 변수 반환
def returnCounter(nouns):
    counter = Counter(nouns)
    return counter


# available_counter 변수 반환
# 사용법 : removeOneLetterNoun(counter)
def removeOneLetterNoun(counter):
    available_counter = Counter({x: counter[x] for x in counter if len(x) > 1})
    return available_counter


def addStopwords(stopwords):
    stopwords = pd.read_csv(
        "https://raw.githubusercontent.com/yoonkt200/FastCampusDataset/master/korean_stopwords.txt").values.tolist()
    jeju_list = ['제주', '제주도', '호텔', '리뷰', '숙소', '여행', '트립']
    for word in jeju_list:
        stopwords.append(word)
    return stopwords


# 정규 표현식 + 불용어 처리
def text_cleaning(text, stopwords):
    hangul = re.compile("[^ ㄱ-ㅣ가-힣]")
    result = hangul.sub('', str(text))
    tagger = Okt()
    nouns = tagger.nouns(result)  # 여기까지 정규표현식 적용
    nouns = [x for x in nouns if len(x) > 1]  # 한글자 키워드 제거
    nouns = [x for x in nouns if x not in stopwords]  # 불용어 처리
    return nouns


def returnVect(text_cleaning):
    from sklearn.feature_extraction.text import CountVectorizer
    vect = CountVectorizer(tokenizer=lambda x: text_cleaning(x))
    return vect


def returnBow_vect(vect, dataframe):
    bow_vect = vect.fit_transform(dataframe['text'].tolist())
    return bow_vect


def returnWord_list(vect):
    word_list = vect.get_feature_names()
    return word_list


def returnCount_list(bow_vect):
    count_list = bow_vect.toarray().sum(axis=0)
    return count_list


def returnWordCountDict(word_list, count_list):
    word_count_dict = dict(zip(word_list, count_list))
    return word_count_dict


# IF-IDF 생성
def returnTFIDF_Vectorizer():
    from sklearn.feature_extraction.text import TfidfTransformer
    tfidf_vectorizer = TfidfTransformer()
    return tfidf_vectorizer


def returnTFIDF_Vect(tfidfVectorizer, bow_vect):
    tf_idf_vect = tfidfVectorizer.fit_transform(bow_vect)
    return tf_idf_vect


# 단어 맵핑
def returnInvertedIdxVectorizer(vect):
    invert_index_vectorizer = {v: k for k, v in vect.vocabulary_.items()}
    return invert_index_vectorizer


# 로지스틱 회귀 분석 시작
# rating 칼럼 이진화
def rating_to_label(rating):
    if rating > 3:
        return 1  # rating값이 3초과이면 (긍정적 단어라면) 값은 1
    else:
        return 0  # rating값이 3미만이면 (부정적 단어라면) 값은 0


# 데이터셋 분리
# x_train, x_test 데이터를 각각 1:1 비율로 조정
# 긍정적 단어의 인덱스
def returnPosSampleIdx(dataframe):
    positive_sample_idx = dataframe[dataframe['y'] == 1].sample(275, random_state=33).index.tolist()
    return positive_sample_idx


# 부정적 단어의 인덱스
def returnNegSampleIdx(dataframe):
    negative_sample_idx = dataframe[dataframe['y'] == 0].sample(275, random_state=33).index.tolist()
    return negative_sample_idx


# 데이터셋 분리
def splitDataSet(positive_sample_idx, negative_sample_idx, tf_idf_vect):
    from sklearn.model_selection import train_test_split
    random_idx = positive_sample_idx + negative_sample_idx
    x = tf_idf_vect[random_idx]
    y = df["y"][random_idx]
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3)
    return x_train, x_test, y_train, y_test


# 모델 학습
def LogisticRegression(x_train, x_test, y_train):
    from sklearn.linear_model import LogisticRegression
    lr = LogisticRegression(random_state=0)
    lr.fit(x_train, y_train)
    y_pred = lr.predict(x_test)
    return lr, y_pred


# 성능평가
def ModelPerformanceEvaluation(y_test, y_pred):
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
    from sklearn.metrics import confusion_matrix
    accuracy_score = accuracy_score(y_test, y_pred)
    precision_score = precision_score(y_test, y_pred)
    recall_score = recall_score(y_test, y_pred)
    f1_score = f1_score(y_test, y_pred)
    confmat = confusion_matrix(y_test, y_pred)
    return accuracy_score, precision_score, recall_score, f1_score, confmat


# 긍정 / 부정 키워드 분석
def keywordCoef(lr):
    import matplotlib.pyplot as plt
    plt.rcParams['figure.figsize'] = [10, 8]
    plt.bar(range(len(lr.coef_[0])), lr.coef_[0])
    plt.show()


# 긍정 단어 상위 n개 추출
def setPosWords(coef_pos_index, invert_index_vectorizer, n):
    pos_words = list()
    for coef in coef_pos_index[:n]:
        pos_words.append(invert_index_vectorizer[coef[1]])
    return pos_words


# 긍정 단어 coef값 상위 n개 추출
def setPosCoefs(coef_pos_index, n):
    pos_coef = list()
    for coef in coef_pos_index[:n]:
        pos_coef.append(coef[0])
    return pos_coef


# 긍정 단어 시각화
def returnPosWordsBarh(setPosWords, setPosCoefs):
    import matplotlib.pyplot as plt
    plt.rc('font', family='gulim')
    plt.rcParams["figure.figsize"] = (12, 9)
    plt.barh(setPosWords, setPosCoefs,
             align='center',
             alpha=0.5,
             color='blue')
    plt.xlabel('words')
    plt.title('count')
    plt.show()


# 부정 키워드 n개 설정
def setNegWords(coef_neg_index, invert_index_vectorizer, n):
    pos_words = list()
    for coef in coef_neg_index[:n]:
        pos_words.append(invert_index_vectorizer[coef[1]])
    return pos_words


# 긍정 단어 coef값 상위 n개 추출
def setNegCoefs(coef_neg_index, n):
    pos_coef = list()
    for coef in coef_neg_index[:n]:
        pos_coef.append(coef[0])
    return pos_coef


# 긍정 단어 시각화
def returnNegWordsBarh(setNegWords, setNegCoefs):
    import matplotlib.pyplot as plt
    plt.rc('font', family='gulim')
    plt.rcParams["figure.figsize"] = (12, 9)
    plt.barh(setNegWords, setNegCoefs,
             align='center',
             alpha=0.5,
             color='blue')
    plt.xlabel('words')
    plt.title('count')
    plt.show()


# 부정적인 단어 관련
## 부정 단어 상위 15개 추출
def setNegWords(coef_neg_index, index_vectorizer, n):
    neg_words = list()
    for coef in coef_neg_index[:n]:
        neg_words.append(index_vectorizer[coef[1]])
    return neg_words


## 부정단어 coef값 상위 15개 추출
def setNegCoefs(coef_neg_index, n):
    neg_coef = list()
    for coef in coef_neg_index[:n]:
        neg_coef.append(coef[0])
    return neg_coef


## 부정적 word / coef의 barhplot 반환
def returnNegWordsBarh(setNegWords, setNegCoefs):
    import matplotlib.pyplot as plt
    plt.rc('font', family='gulim')
    plt.rcParams["figure.figsize"] = (12, 9)
    plt.barh(setNegWords, setNegCoefs, alpha=0.5, align='center', color='red')
    plt.xlabel('words')
    plt.title('count')
    plt.show()

def returnWordcloud(word_count_dict):
    import matplotlib.pyplot as plt
    from wordcloud import WordCloud
    wordcloud = WordCloud(font_path='C:/Windows/Fonts/gulim.ttc',
                          width=500, height=500,
                          background_color = "white",
                          max_font_size = 150,
                          min_font_size = 7,
                          margin = 3).generate_from_frequencies(word_count_dict)
    plt.figure(figsize=(15, 15))
    plt.imshow(wordcloud, interpolation="lanczos")
    plt.axis("off")
    plt.show()