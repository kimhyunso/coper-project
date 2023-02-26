import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import re
import json

from konlpy.tag import Okt
from collections import Counter
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer

import warnings

warnings.filterwarnings("ignore")


# result 변수 반환
def apply_regular_expression(text: str) -> str:
    """
    정규표현식을 적용한 str 객체를 반환한다.

    파라미터:
    --------

    text : 문장 한 줄을 인자로 받아, 정규표현식을 적용한다.

    사용예:

    apply_regular_expression(df['text'][숫자])
    """
    import re

    hangul = re.compile("[^ ㄱ-ㅣ가-힣|A-Z|a-z]")
    result = hangul.sub("", text)
    return result


# nouns 변수 반환
def nouns(sentence: str) -> list:
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
def counter(nouns: list) -> list:
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
def remove_one_letter_noun(counter: list) -> list:
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
def stopwords(stopwords_path: str, stopwords_list: list = [""]) -> list:
    """
    원하는 불용어가 추가된 불용어 리스트를 반환한다.

    파라미터:

    stopwords_txt = 불용어 리스트(korean_stopwords.txt)를 txt파일로 받는다.
    stopwords_list : stopwords에 불용어를 추가한다. 이 인자는 반드시 리스트로 받아야 한다.

    사용예제:

    add_stopwords(['이거', '영상', '유튜브'])
    """
    infile = open(stopwords_path, "r", encoding="utf-8").readlines()
    stopwords = []
    for line in infile:
        stopwords.append(line.replace("\n", ""))

    if stopwords_list:
        for word in stopwords_list:
            stopwords.append(word)
    return stopwords


# 정규 표현식 + 불용어 처리
def text_cleaning(text: str, stopwords: list) -> list:
    """
    정규표현식과 불용어를 동시에 처리하고, 명사 단위로 쪼갠 리스트를 반환한다.
    즉, apply_regular_expression + add_stopwords + return_nouns

    파라미터

    text : 텍스트 문장 한 줄을 인자로 받는다.

    stopwords : 불용어 리스트를 인자로 받는다.


    사용예제:

    > text_cleaning(df['text'][0])
    """
    hangul = re.compile("[^ ㄱ-ㅣ가-힣]")
    result = hangul.sub("", str(text))
    tagger = Okt()
    nouns = tagger.nouns(result)  # 여기까지 정규표현식 적용
    nouns = [x for x in nouns if len(x) > 1]  # 한글자 키워드 제거
    nouns = [x for x in nouns if x not in stopwords]  # 불용어 처리
    return nouns


def invert_text_to_vect(words: list) -> list:
    result = []
    word_document_path = (
        "C:/Users/TECH2_07/Desktop/이도원 프로젝트 폴더/자료/KnuSentiLex/SentiWord_info - 복사본.json"
    )
    with open(word_document_path, encoding="utf-8", mode="r") as f:
        data = json.load(f)

    for i in range(0, len(data)):
        for word in words:
            if word == data[i]["word"]:
                result.append(int(data[i]["polarity"]))
    return result


# 긍정적 단어의 인덱스
def pos_word_index(dataframe: pd.DataFrame, sample_size: int) -> list:
    """
    긍정적인 단어들의 인덱스를 반환한다.

    파라미터:

    dataframe : 반환할 단어가 포함된 DataFrame을 인자로 받는다.
    sample_size : 추출할 샘플의 크기를 지정하는 Integer값을 인자로 받는다.

    사용예제:

    pos_word_index(df, df.y.value_counts()[0])
    """
    positive_sample_idx = (
        dataframe[dataframe["y"] == 2]
        .sample(sample_size, random_state=33)
        .index.tolist()
    )
    return positive_sample_idx


# 부정적 단어의 인덱스
def neg_word_index(dataframe: pd.DataFrame, sample_size: int) -> list:
    negative_sample_idx = (
        dataframe[dataframe["y"] == 1]
        .sample(sample_size, random_state=33)
        .index.tolist()
    )
    return negative_sample_idx


# hashtag column에서 해시태그만 남기기
def extract_hashtags(text: str, name: str = "") -> str:
    """
    hashtag column에서 공백을 추가해서 해시태그만 반환한다.

    파라미터:

    text : 해시태그만 추출할 문자열을 인자로 받는다.
    name : 추출할 해시태그에서 자기자신(크리에이터 본인)의 해시태그는 삭제한다.
    기본(Default)는 없고, 해시태그 이름의 뒤에는 반드시 공백을 추가해야한다.

    사용예제:

    > test(df.hashtag[0])
    > test(df.hashtag[0], "#오킹 ")
    > df.hashtag.apply(test)
    """
    tags = " ".join(re.findall("#\w+", text))
    result = re.sub(name, "", tags)
    return result


def extract_human_hashtag(text: str, df: pd.DataFrame, human_list_path: str = ""
) -> str:
    """
    df의 hashtag column에서 사람 태그만 남긴다.

    파라미터:

    text : DataFrame의 hashtag column을 인자로 받는다.
    df : 해당 DataFrame을 인자로 받는다. 이때, 반드시 받을 인자는 초기의 df이다.
    human_list_path : 인물리스트에 대한 파일 위치를 나타내는 문자열 값을 인자로 받는다. 기본값은 ("")없음.

    사용예제:

    > prep.extract_one_hash(df.hashtag, df)
    > new_df.hashtag.apply(lambda text: prep.extract_one_hash(text, new_df))
    """
    result = []
    target_list = text.split(" ")
    most_list = hashtag_list(df)

    if human_list_path:
        human_array = pd.read_csv(
            human_list_path, encoding="utf-8", engine="python", sep="\t"
        ).values

    for hashtag in target_list:
        if human_list_path:  # human_list_path에 하나라도 있으면
            if hashtag in most_list and hashtag in human_array:
                result.append(hashtag)
        else:  # human_list_path에 하나도 없으면
            if hashtag in most_list:
                result.append(hashtag)
    if len(result) == 0:
        result.append("None")
    return " ".join(result)


def hashtag_list(df: pd.DataFrame) -> list:
    """
    DataFrame의 hashtag column을 입력받아, 해시태그 단위로 분리된 문자열이 담긴 리스트를 반환한다.

    파라미터:

    series : DataFrame의 column 중에서 동영상의 해시태그가 담긴 column을 Series 객체로 받는다.

    사용예제:

    > hashtag_list(df.hashtag)
    """
    hashtag_list = []
    for text in df.hashtag:
        hashtag_list += text.split(" ")
    return hashtag_list


def most_used_hashtag_list(df: pd.DataFrame) -> list:
    """
    평균 이상으로 사용된 해시태그의 리스트를 반환한다.

    파라미터:
    df : hashtag 열이 존재하는 DataFrame을 인자로 받는다. 이때, hashtag은 해시태그만 존재하여야 한다.

    사용예제:
    > most_used_hashtag_list(df)
    """
    hashtag_count = Counter(hashtag_list(df))
    hashtag_count_1 = list(hashtag_count.values())
    most_used_hash_list = list(
        {
            key
            for key, value in hashtag_count.items()
            if value > np.mean(hashtag_count_1)
        }
    )
    return most_used_hash_list


def most_used_hashtag_df(df: pd.DataFrame) -> pd.DataFrame:
    """
    평균 이상으로 사용된 해시태그가 포함된 row의 DataFrame을 반환한다.

    파라미터:

    df : hashtag 열이 존재하는 DataFrame을 인자로 받는다. 이때, hashtag은 해시태그만 존재하여야 한다.

    사용예제:
    > prep.most_used_hashtag_df(df)
    > (prep.most_used_hashtag_df(df).hashtag.str.contains('#방랑화가이병건') == True).sum()
    """
    many_hashtag_df = df.copy().drop(df.index, axis=0)

    for most_used_hashtag in most_used_hashtag_list(df):
        df_desc_bool = df.hashtag.str.contains(most_used_hashtag)
        temp_df = df.loc[df_desc_bool]
        many_hashtag_df = pd.concat([many_hashtag_df, temp_df])

    many_hashtag_df.drop_duplicates(inplace=True)
    return many_hashtag_df


def view_like_count_and_df_index(df:pd.DataFrame, represent:str="mean", tag_name:str=""
)-> tuple:
    """
    views_count, like_count, df_index를 각각 반환합니다.

    인수:
    - df: 해시태그, 보기 및 좋아요를 포함한 비디오 정보가 포함된 pandas.DataFrame.
    - represent: 각 해시태그의 카운트를 나타내는 방법을 지정하는 문자열입니다.
        - "mean": 각 해시태그에 대한 뷰 및 좋아요 카운트의 평균을 반환합니다.
        - "sum": 각 해시태그에 대한 뷰와 좋아요 개수의 합계를 반환합니다.
        - "median": 각 해시태그에 대한 뷰 및 유사 카운트의 중위수를 반환합니다.
    - tag_name: 대푯값에 대한 기준인 해시태그를 설정합니다.

    반환:
    - 해시태그를 인덱스로 사용하고 비디오 인덱스를 열로 사용하는 DataFrame(각 요소가 있는 곳)
    해당 해시태그 및 비디오 인덱스에 대한 보기 또는 좋아요 수를 나타냅니다.
    """
    if tag_name:
        df_index = df.loc[df.hashtag==tag_name].index
    else:
        raise Exception("tag_name에 정확한 값을 입력하시오.")

    if represent == "sum":
        views_count = df.loc[df_index, "views_count"].sum()
        like_count = df.loc[df_index, "like_count"].sum()
    elif represent == "mean":
        views_count = np.mean(df.loc[df_index, "views_count"])
        like_count = np.mean(df.loc[df_index, "like_count"])
    elif represent == "median":
        views_count = np.median(df.loc[df_index, "views_count"])
        like_count = np.median(df.loc[df_index, "like_count"])
    return views_count, like_count, df_index


def add_drop_row(df:pd.DataFrame, df_index:list, add_list:list) -> pd.DataFrame:
    """
    DataFrame에서 특정 인덱스의 행을 제거하고, 새로운 행을 추가하여 반환하는 함수.

    Args:
    - df (pd.DataFrame): 특정 행을 제거할 DataFrame
    - df_index (list): 제거할 DataFrame 행의 인덱스 리스트
    - add_list (list): 추가할 행의 정보를 담고 있는 리스트. 순서대로 video_id, category_id, category_name, title, views_count, like_count, uploaded_at, hashtag 정보를 담고 있음

    Returns:
    - new_df (pd.DataFrame): 새롭게 업데이트된 DataFrame.
    """

    # 인덱스에 해당하는 행을 제거한 새로운 DataFrame 생성
    new_df = df.drop(df_index)

    # 추가할 행의 정보를 담은 딕셔너리 생성
    new_row_dict = {
        'video_id' : add_list[0],
        'category_id' : add_list[1],
        'category_name' : add_list[2],
        'title' : add_list[3],
        'views_count' : add_list[4],
        'like_count' : add_list[5],
        'uploaded_at' : add_list[6],
        'hashtag' : add_list[7],
    }

    # 새로운 행 추가
    new_df = new_df.append(new_row_dict, ignore_index=True)

    # 새롭게 업데이트된 DataFrame 반환
    return new_df


def automatize_human_hash_df(df:pd.DataFrame, s_tag_name:str="", w_tag_name:str="", category_id:int=0
) -> pd.DataFrame:
    """
    이 함수는 pd.DataFrame 객체와 함께 인자로 받은 해시태그 이름과 카테고리 ID를 이용하여 새로운 로우를 추가하는 함수입니다.

    함수 인자
    - df : pd.DataFrame 새로운 로우를 추가할 데이터프레임
    - s_tag_name : str 검색에 사용된 해시태그 이름, 기본값은 빈 문자열 ""
    - w_tag_name : str 추가할 해시태그 이름. 해시태그로 시작해야하며, 이 값이 변경되면 새로운 로우의 hashtag값도 변경됩니다.
    - category_id : int 추가할 로우의 카테고리 ID. int형으로 지정되어야 합니다.

    함수 동작
    - s_tag_name을 이용하여 조회수와 좋아요 수 그리고 인덱스를 구합니다.
    - w_tag_name과 category_id로부터 새로운 로우를 생성합니다. 구한 인덱스를 이용하여 데이터프레임에서 해당 로우를 삭제하고, 새로운 로우를 추가합니다.
    
    이후, 변경된 데이터프레임을 반환합니다.
    
    함수 예외
    - w_tag_name이 빈 문자열이거나, category_id가 지정되지 않은 경우 Exception 예외를 발생합니다.
    """
    if not w_tag_name or not category_id or w_tag_name[0] != "#":
        raise Exception("해시태그나 카테고리 ID 값을 확인해주세요.")
    
    views_count, like_count, df_index = view_like_count_and_df_index(df, tag_name=s_tag_name)
    add_list = [f'{w_tag_name}_video', category_id, 'Entertainment', f'{w_tag_name}_title', views_count, like_count, f'{w_tag_name}_uploaded_at', w_tag_name]
    new_df = add_drop_row(df, df_index, add_list)
    return new_df


