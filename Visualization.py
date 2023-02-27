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


# barplot 시각화
# 긍정 / 부정 키워드 분석
def keywordCoef(lr):
    import matplotlib.pyplot as plt

    plt.rcParams["figure.figsize"] = [10, 8]
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

    plt.rc("font", family="gulim")
    plt.rcParams["figure.figsize"] = (12, 9)
    plt.barh(setPosWords, setPosCoefs, align="center", alpha=0.5, color="blue")
    plt.xlabel("words")
    plt.title("count")
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

    plt.rc("font", family="gulim")
    plt.rcParams["figure.figsize"] = (12, 9)
    plt.barh(setNegWords, setNegCoefs, align="center", alpha=0.5, color="blue")
    plt.xlabel("words")
    plt.title("count")
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

    plt.rc("font", family="gulim")
    plt.rcParams["figure.figsize"] = (12, 9)
    plt.barh(setNegWords, setNegCoefs, alpha=0.5, align="center", color="red")
    plt.xlabel("words")
    plt.title("count")
    plt.show()


def returnWordcloud(word_count_dict):
    import matplotlib.pyplot as plt
    from wordcloud import WordCloud

    wordcloud = WordCloud(
        font_path="C:/Windows/Fonts/gulim.ttc",
        width=500,
        height=500,
        background_color="white",
        max_font_size=150,
        min_font_size=7,
        margin=3,
    ).generate_from_frequencies(word_count_dict)
    plt.figure(figsize=(15, 15))
    plt.imshow(wordcloud, interpolation="lanczos")
    plt.axis("off")
    plt.show()


def appear_video_month_comment_count(df:pd.DataFrame, video_id:str) -> None:
    """
    주어진 DataFrame에서 video_id가 주어진 video_id와 일치하는 댓글 데이터를 가져와서
    시간순으로 정렬하고, 월별 댓글 등록 수와 월별 누적 댓글수를 그래프로 나타내는 함수입니다.
    
    Parameters
    ----------
    df : pandas.DataFrame
        댓글 데이터를 담고 있는 DataFrame
    video_id : str
        댓글을 확인하고자 하는 동영상의 ID
        
    Returns
    -------
    None
        월간 댓글 등록 수와 누적 댓글수를 시각화한 그래프를 출력합니다.
    """
       # video_id와 일치하는 동영상 댓글 데이터만 추출합니다.
    video_df = df.loc[df.video_id == video_id]
    # 댓글 등록 시간을 기준으로 오름차순 정렬합니다.
    video_df = video_df.sort_values(by="created_at", ascending=True).reset_index(drop=True)
    # 댓글 등록 시간을 기준으로 다시 오름차순 정렬합니다.
    video_df_1 = video_df.copy()

    # 월별 댓글 등록 수를 구합니다.
    video_df_month = video_df_1.created_at.dt.month.unique().tolist()
    video_df_month = list(map(str, video_df_month))
    video_df_data = video_df_1.created_at.dt.month.value_counts().values.tolist()

    # 그래프 사이즈를 조절합니다.
    plt.rcParams["figure.figsize"] = (12, 9)

    # 첫 번째 subplot에 월별 댓글 등록 수를 나타내는 선 그래프를 그립니다.
    plt.subplot(1, 2, 1)
    plt.title(f"{video_df_1.video_id[0]}의 월간 댓글 등록 수")
    plt.xlabel("월(Month)")
    plt.ylabel("댓글 등록 수")
    sns.lineplot(x=video_df_month, y=video_df_data, label="댓글 등록 수")

    # 두 번째 subplot에 월별 누적 댓글수를 나타내는 선 그래프를 그립니다.
    # 누적 댓글수는 이전 월의 누적 댓글수에 현재 월의 댓글 등록 수를 더한 값으로 계산합니다.
    new_video_df_data = video_df_data.copy()

    for idx, data in enumerate(video_df_data):
        if idx == 0:
            continue
        new_video_df_data[idx] = new_video_df_data[idx] + new_video_df_data[idx-1]

    plt.subplot(1, 2, 2)
    plt.title(f"{video_df_1.video_id[0]}의 월간 누적 댓글수")
    plt.xlabel("월(Month)")
    plt.ylabel("댓글 등록 수")
    sns.lineplot(x=video_df_month, y=new_video_df_data, label="누적 댓글수", color="red")

    plt.show()


def show_boxplot(df:pd.DataFrame, col_x_y:tuple, graph_size:tuple=(30, 20)):
    """
    입력된 데이터프레임과 컬럼 리스트를 바탕으로 boxplot을 그리는 함수.
    - df: boxplot을 그리기 위한 데이터프레임
    - col_x_y:tuple: boxplot의 인자인 x, y값을 지정하기 위한 tuple. 튜플 내부에는 column을 지정하는 str 데이터가 존재해야 한다.
    - graph_size: 그래프의 크기(가로, 세로)를 설정하는 튜플. 기본값(Default)은 (30, 20)
    """
    # 그래프의 사이즈를 설정합니다.
    plt.rcParams['figure.figsize'] = graph_size
    # 폰트 설정을 합니다.
    plt.rc('font', family='Malgun Gothic')

    # seaborn 라이브러리의 boxplot을 사용하여 그래프를 그립니다.
    sns.boxplot(data=df, x=df[f"{col_x_y[0]}"], y=df[f"{col_x_y[1]}"])
    # 그래프 요소들이 서로 겹치지 않도록 간격을 조절합니다.
    plt.tight_layout()
    # x축 라벨의 폰트 크기를 설정합니다.
    plt.xticks(fontsize = 8.5)
    # 그래프를 화면에 출력합니다.
    plt.show()