import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import re

from konlpy.tag import Okt
from collections import Counter
from wordcloud import WordCloud
from sklearn.feature_extraction.text import CountVectorizer

import warnings

warnings.filterwarnings("ignore")



# result 변수 반환
def apply_regular_expression(text:str) -> str:
    """
    정규표현식을 적용한 str 객체를 반환한다.

    파라미터

    text : 문장 한 줄을 인자로 받아, 정규표현식을 적용한다.

    사용예

    apply_regular_expression(df['text'][숫자])
    """
    import re
    hangul = re.compile('[^ ㄱ-ㅣ가-힣]')
    result = hangul.sub('', text)
    return result


# nouns 변수 반환
def return_nouns(sentence:str) -> list:
    """
    문장 한 줄을 명사(nouns) 단위로 분해한 리스트를 반환한다. 
    예를 들어, '여행에 집중할수 있게 편안한 휴식을 제공하는 호텔이었습니다' 를 입력 받으면
    ['여행', '집중', '휴식', '제공', '호텔'] 을 반환한다.

    파라미터

    sentence : 문장 한 줄을 인자로 받는다.

    사용예

    return_nouns(df['text'][0])
    > ['여행', '집중', '휴식', '제공', '호텔', '위치', '선정', '또한', '청소', '청결', '상태']

    nouns_tagger.nouns(apply_regular_expression("".join(sentence['text'].tolist())))
    > [ '스테이', '위치', '신라', ~~~~~ '스타벅스', '번화가', '전',]
    """
    from konlpy.tag import Okt
    nouns_tagger = Okt()
    nouns = nouns_tagger.nouns(apply_regular_expression(sentence))
    return nouns


# counter 변수 반환
def return_counter(nouns:list) -> list:
    """
    명사 리스트에서 각 명사의 갯수가 몇개인지 반환한다.
    반환값은 기본적으로 list이지만 list 내부는 ('명사', 갯수)의 튜플 형태이다.
    자세한 건 사용예 참조.
    
    파라미터

    nouns : 명사 리스트를 인자로 받는다.

    사용예

    return_counter(nouns).most_common(10)
    > 
    [('호텔', 803),
    ('수', 498),
    ('것', 436),
    ('방', 330),
    ('위치', 328),
    ('우리', 327),
    ('곳', 320),
    ('공항', 307),
    ('직원', 267),
    ('매우', 264)]
    """
    counter = Counter(nouns)
    return counter


# 사용법 : remove_one_letter_noun(counter)
def remove_one_letter_noun(counter:list) -> list:
    """
    한 글자 명사를 제거한 list 객체를 반환한다. 
    리스트 내포의 결과값 예시

    > x : counter[x] = '타일' : 3

    파라미터

    counter : Collections.Counter 메소드에 의해 처리된 list값을 인자로 받는다.
    즉, '단어' : 갯수 튜플로 감싸진 리스트 형태이다.

    사용법
    remove_one_letter_noun(counter).most_common(10)
    > 
    [('호텔', 803),
    ('위치', 328),
    ('우리', 327),
    ('공항', 307),
    ('직원', 267),
    ('매우', 264),
    ('가격', 245),
    ('객실', 244),
    ('시설', 215),
    ('제주', 192)]
    """
    available_counter = Counter({x: counter[x] for x in counter if len(x) > 1})
    return available_counter


# 불용어 리스트 추가
# 사용법 : add_stopwords(['이거', '저거', '호텔'])
def add_stopwords(stopwords_txt:object, stopwords_list:list) -> list:
    """
    원하는 불용어가 추가된 불용어 리스트를 반환한다.
    
    파라미터
    stopwords_txt = 불용어 리스트(korean_stopwords.txt)를 txt파일로 받는다.
    stopwords_list : stopwords에 불용어를 추가한다. 이 인자는 반드시 리스트로 받아야 한다.

    사용예)
    add_stopwords(['이거', '영상', '유튜브'])
    """
    stopwords = pd.read_csv(stopwords_txt).values.tolist()
    for word in stopwords_list:
        stopwords.append(word)
    return stopwords


# 정규 표현식 + 불용어 처리
def text_cleaning(text:str, stopwords:list) -> list:
    """
    정규표현식과 불용어를 동시에 처리하고, 명사 단위로 쪼갠 리스트를 반환한다.
    즉, apply_regular_expression + add_stopwords + return_nouns

    파라미터
    
    text : 텍스트 문장 한 줄을 인자로 받는다.

    stopwords : 불용어 리스트를 인자로 받는다.


    사용예

    $ text_cleaning(df['text'][0])

    > ['집중', '휴식', '제공', '위치', '선정', '또한', '청소', '청결', '상태']
    """
    hangul = re.compile("[^ ㄱ-ㅣ가-힣]")
    result = hangul.sub('', str(text))
    tagger = Okt()
    nouns = tagger.nouns(result)  # 여기까지 정규표현식 적용
    nouns = [x for x in nouns if len(x) > 1]  # 한글자 키워드 제거
    nouns = [x for x in nouns if x not in stopwords]  # 불용어 처리
    return nouns


# 카운트 기반 벡터화
def vect(text_cleaning:list) -> object:
    """
    CountVectorizer의 인자인 tokenizer을 지정하기 위한 메소드를 입력받아, 생성된 CountVectorizer 객체를 반환한다.
    이 객체를 훈련시켜(fit_transform) 단어들만 뽑아내거나, 해당 위치의 단어가 몇 번 등장했는지도 알 수 있다.

    
    파라미터

    text_cleaning : CountVectorizer 메소드의 인자 tokenizer를 지정하기 위한 인자를 받는다.

    사용예

    from sklearn.feature_extraction.text import CountVectorizer
    
    return_vect(text_cleaning)
    
    > CountVectorizer()
    """
    vect = CountVectorizer(tokenizer=lambda x: text_cleaning(x))
    return vect


def bow_vect(vect:object, dataframe:object) -> object:
    """
    
    """
    bow_vect = vect.fit_transform(dataframe['text'].tolist())
    return bow_vect


def word_list(vect):
    word_list = vect.get_feature_names()
    return word_list


def count_list(bow_vect):
    count_list = bow_vect.toarray().sum(axis=0)
    return count_list


def word_count_dict(word_list, count_list):
    word_count_dict = dict(zip(word_list, count_list))
    return word_count_dict


# IF-IDF 생성
def return_TFIDF_vectorizer():
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