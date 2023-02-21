import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import re
import json

from konlpy.tag import Okt
from collections import Counter
from wordcloud import WordCloud
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer

import warnings

warnings.filterwarnings("ignore")

# result 변수 반환
def apply_regular_expression(text:str) -> str:
    """
    정규표현식을 적용한 str 객체를 반환한다.

    파라미터: 
    --------

    text : 문장 한 줄을 인자로 받아, 정규표현식을 적용한다.

    사용예: 

    apply_regular_expression(df['text'][숫자])
    """
    import re
    hangul = re.compile('[^ ㄱ-ㅣ가-힣|A-Z|a-z]')
    result = hangul.sub('', text)
    return result


# nouns 변수 반환
def nouns(sentence:str) -> list:
    """
    문장 한 줄을 명사(nouns) 단위로 분해한 리스트를 반환한다. 
    예를 들어, '여행에 집중할수 있게 편안한 휴식을 제공하는 호텔이었습니다' 를 입력 받으면
    ['여행', '집중', '휴식', '제공', '호텔'] 을 반환한다.

    파라미터: 

    sentence : 문장 한 줄을 인자로 받는다.

    사용예제:

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
def counter(nouns:list) -> list:
    """
    명사 리스트에서 각 명사의 갯수가 몇개인지 반환한다.
    반환값은 기본적으로 list이지만 list 내부는 ('명사', 갯수)의 튜플 형태이다.
    자세한 건 사용예 참조.
    
    파라미터: 

    nouns : 명사 리스트를 인자로 받는다.

    사용예제:

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

    파라미터:

    counter : Collections.Counter 메소드에 의해 처리된 list값을 인자로 받는다.
    즉, '단어' : 갯수 튜플로 감싸진 리스트 형태이다.

    사용예제:

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
def stopwords(stopwords_path:str, stopwords_list:list) -> list:
    """
    원하는 불용어가 추가된 불용어 리스트를 반환한다.
    
    파라미터:

    stopwords_txt = 불용어 리스트(korean_stopwords.txt)를 txt파일로 받는다.
    stopwords_list : stopwords에 불용어를 추가한다. 이 인자는 반드시 리스트로 받아야 한다.

    사용예제:

    add_stopwords(['이거', '영상', '유튜브'])
    """
    infile = open(stopwords_path, 'r', encoding='utf-8').readlines()
    stopwords = []
    for line in infile:
        stopwords.append(line.replace('\n', ''))

    if stopwords_list:
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


    사용예제:

    $ text_cleaning(df['text'][0])
    """
    hangul = re.compile("[^ ㄱ-ㅣ가-힣]")
    result = hangul.sub('', str(text))
    tagger = Okt()
    nouns = tagger.nouns(result)  # 여기까지 정규표현식 적용
    nouns = [x for x in nouns if len(x) > 1]  # 한글자 키워드 제거
    nouns = [x for x in nouns if x not in stopwords]  # 불용어 처리
    return nouns


def invert_text_to_vect(words:list) -> list:
    result = []
    word_document_path = "C:/Users/TECH2_07/Desktop/이도원 프로젝트 폴더/자료/KnuSentiLex/SentiWord_info - 복사본.json"
    with open(word_document_path, encoding='utf-8', mode='r') as f:
        data = json.load(f)
    
    for i in range(0, len(data)):
        for word in words:
            if word == data[i]['word']:
                result.append(int(data[i]['polarity']))
    return result
