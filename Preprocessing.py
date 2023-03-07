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
from sklearn.preprocessing import LabelEncoder

import warnings
# from tqdm import tqdm

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
    nouns = tagger.nouns(result.strip())  # 여기까지 정규표현식 적용
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


# tags column에서 해시태그만 남기기
def extract_tags(text: str, name: str = "") -> str:
    """
    tags column에서 공백을 추가해서 해시태그만 반환한다.

    파라미터:

    text : 해시태그만 추출할 문자열을 인자로 받는다.
    name : 추출할 해시태그에서 자기자신(크리에이터 본인)의 해시태그는 삭제한다.
    기본(Default)는 없고, 해시태그 이름의 뒤에는 반드시 공백을 추가해야한다.

    사용예제:

    > test(df.tags[0])
    > test(df.tags[0], "#오킹 ")
    > df.tags.apply(test)
    """
    tags = " ".join(re.findall("#\w+", text))
    result = re.sub(name, "", tags)
    return result


def extract_human_tags(
    text: str, df: pd.DataFrame, col_name: str, human_list_path: str = ""
) -> str:
    """
    df의 tags column에서 사람 태그만 남긴다.

    파라미터:

    text : DataFrame의 tags column을 인자로 받는다.
    df : 해당 DataFrame을 인자로 받는다. 이때, 반드시 받을 인자는 초기의 df이다.
    human_list_path : 인물리스트에 대한 파일 위치를 나타내는 문자열 값을 인자로 받는다. 기본값은 ("")없음.

    사용예제:

    > prep.extract_one_hash(df.tags, df)
    > new_df.tags.apply(lambda text: prep.extract_one_hash(text, new_df))
    """

    result = []
    target_list = text.split(" ")
    most_list = str_list(df, col_name)

    if human_list_path:
        human_array = pd.read_csv(
            human_list_path, encoding="utf-8", engine="python", sep="\t"
        ).values

    for tags in target_list:
        if human_list_path:  # human_list_path에 하나라도 있으면
            if tags in most_list and tags in human_array:
                result.append(tags)
        else:  # human_list_path에 하나도 없으면
            if tags in most_list:
                result.append(tags)
    if len(result) == 0:
        result.append("None")
    return " ".join(result)


def str_list(df: pd.DataFrame, col_name: str) -> list:
    """
    DataFrame의 tags column을 입력받아, 해시태그 단위로 분리된 문자열이 담긴 리스트를 반환한다.

    파라미터:

    series : DataFrame의 column 중에서 동영상의 해시태그가 담긴 column을 Series 객체로 받는다.

    사용예제:

    > str_list(df, col_name)
    """
    tags_list = []
    for text in df[col_name]:
        # text = text.split()
        tags_list += text
    return tags_list


def most_used_str_list(df: pd.DataFrame, col_name: str) -> list:
    """
    평균 이상으로 사용된 해시태그의 리스트를 반환한다.

    파라미터:
    df : tags 열이 존재하는 DataFrame을 인자로 받는다. 이때, tags은 해시태그만 존재하여야 한다.

    사용예제:
    > most_used_str_list(df, "tags")
    """
    tags_count = Counter(str_list(df, col_name))
    tags_count_1 = list(tags_count.values())
    most_used_hash_list = list(
        {key for key, value in tags_count.items() if value > np.mean(tags_count_1)}
    )
    return most_used_hash_list


def most_used_str_df(df: pd.DataFrame, col_name: str) -> pd.DataFrame:
    """
    평균 이상으로 사용된 해시태그가 포함된 row의 DataFrame을 반환한다.

    파라미터:

    df : tags 열이 존재하는 DataFrame을 인자로 받는다. 이때, tags은 해시태그만 존재하여야 한다.

    사용예제:
    > prep.most_used_tags_df(df)
    > (prep.most_used_tags_df(df).tags.str.contains('#방랑화가이병건') == True).sum()
    """
    many_tags_df = df.copy().drop(df.index, axis=0)

    for most_used_tags in most_used_str_list(df, col_name):
        df_desc_bool = df[col_name].str.contains(most_used_tags)
        temp_df = df.loc[df_desc_bool]
        many_tags_df = pd.concat([many_tags_df, temp_df])

    many_tags_df.drop_duplicates(inplace=True)
    return many_tags_df


def view_like_count_and_df_index(
    df: pd.DataFrame, represent: str = "mean", tag_name: str = ""
) -> tuple:
    """
    해시태그 단위로, 조회수와 좋아요 수에 대한 각각의 대표값이 담긴 views_count, like_count와
    해당 해시태그가 일치하는 DataFrame의 인덱스 df_index를 각각 반환합니다.

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
        df_index = df.loc[df.tags == tag_name].index
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


def add_drop_row(df: pd.DataFrame, df_index: list, add_list: list) -> pd.DataFrame:
    """
    DataFrame에서 특정 인덱스의 행을 제거하고, 새로운 행을 추가하여 반환하는 함수.

    Args:
    - df (pd.DataFrame): 특정 행을 제거할 DataFrame
    - df_index (list): 제거할 DataFrame 행의 인덱스 리스트
    - add_list (list): 추가할 행의 정보를 담고 있는 리스트. 순서대로 video_id, category_id, category_name, title, views_count, like_count, uploaded_at, tags 정보를 담고 있음

    Returns:
    - new_df (pd.DataFrame): 새롭게 업데이트된 DataFrame.
    """

    # 인덱스에 해당하는 행을 제거한 새로운 DataFrame 생성
    new_df = df.drop(df_index)

    # 추가할 행의 정보를 담은 딕셔너리 생성
    new_row_dict = {
        "video_id": add_list[0],
        "category_id": add_list[1],
        "category_name": add_list[2],
        "title": add_list[3],
        "views_count": add_list[4],
        "like_count": add_list[5],
        "uploaded_at": add_list[6],
        "tags": add_list[7],
    }

    # 새로운 행 추가
    new_df = new_df.append(new_row_dict, ignore_index=True)

    # 새롭게 업데이트된 DataFrame 반환
    return new_df


def automatize_human_hash_df(
    df: pd.DataFrame, s_tag_name: str = "", w_tag_name: str = "", category_id: int = 0
) -> pd.DataFrame:
    """
    이 함수는 pd.DataFrame 객체와 함께 인자로 받은 해시태그 이름과 카테고리 ID를 이용하여 새로운 로우를 추가하는 함수입니다.

    함수 인자
    - df : pd.DataFrame 새로운 로우를 추가할 데이터프레임
    - s_tag_name : str 검색에 사용된 해시태그 이름, 기본값은 빈 문자열 ""
    - w_tag_name : str 추가할 해시태그 이름. 해시태그로 시작해야하며, 이 값이 변경되면 새로운 로우의 tags값도 변경됩니다.
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

    views_count, like_count, df_index = view_like_count_and_df_index(
        df, tag_name=s_tag_name
    )
    add_list = [
        f"{w_tag_name}_video",
        category_id,
        "Entertainment",
        f"{w_tag_name}_title",
        views_count,
        like_count,
        f"{w_tag_name}_uploaded_at",
        w_tag_name,
    ]
    new_df = add_drop_row(df, df_index, add_list)
    return new_df


def get_video_statistics(df_videos: pd.DataFrame, df_comments: pd.DataFrame) -> tuple:
    """
    비디오 데이터프레임(df_videos)과 댓글 데이터프레임(df_comments)을 이용하여 각 비디오의 댓글 수, 조회수, 좋아요 수에 대한 딕셔너리를 생성합니다.

    Args:
    - df_videos (pandas.DataFrame) : 비디오 데이터프레임입니다. (video_id, title, channel_id, channel_title, category_id, publish_date, tags, views_count, likes, dislikes, description)
    - df_comments
      (pandas.DataFrame) : 댓글 데이터프레임입니다. (comment_id, author, video_id, content, publish_date)

    Returns:
    - tuple : 조회수, 좋아요 수, 댓글 수에 대한 딕셔너리를 반환합니다.
    """

    # 각 비디오의 댓글 수, 조회수, 좋아요 수를 계산합니다.
    comment_count = df_comments.video_id.value_counts().values
    video_list = df_videos.video_id.unique()
    views_comment = df_videos.views_count.unique()
    views_like = df_videos.like_count.unique()

    # 각 비디오의 id를 key로 하고 댓글 수, 조회수, 좋아요 수를 value로 하는 딕셔너리를 생성합니다.
    dict_video_comment = dict(zip(video_list, comment_count))  # {비디오_id : 댓글 수}
    dict_video_views = dict(zip(video_list, views_comment))  # {비디오_id : 조회수}
    dict_video_like = dict(zip(video_list, views_like))  # {비디오_id : 좋아요 수}

    return dict_video_views, dict_video_like, dict_video_comment


def encoded_human_dict(df: pd.DataFrame) -> dict:
    """
    DataFrame 형태로 들어온 데이터의 "tags" column을 가지고 각 태그들을 인코딩하여
    dictionary 형태로 반환하는 함수입니다.

    Args:
    - df_videos (pandas.DataFrame) : 'tags' column에 사람에 대한 해시태그만 남겨진 데이터프레임입니다.

    Returns:
    - dict : {해시태그 : 숫자} 쌍으로 구성된 딕셔너리를 반환한다.
    """
    # 빈 리스트 생성
    new_list = []
    # 데이터프레임을 복사하여 index reset
    df_copy = df.copy().reset_index(drop=True)
    # 새로운 열 "encoded_tags"를 NaN 값으로 추가
    df_copy["encoded_tags"] = np.nan

    # 각 행의 "tags" column에서 unique한 값들에 대해 loop 실행
    for human in df_copy.tags.unique():
        # "tags" column에서 공백을 기준으로 나누어서 리스트에 추가
        temp_list = human.split(" ")
        new_list += temp_list

    # 중복 제거
    new_list = list(set(new_list))

    # LabelEncoder를 이용하여 각 태그 인코딩
    encoder = LabelEncoder()
    encoded_list = encoder.fit_transform(new_list)

    # 딕셔너리 형태로 변환
    list_dict = dict(zip(new_list, encoded_list))

    # 딕셔너리 반환
    return list_dict


def encoded_human_df(df: pd.DataFrame, creator: str) -> pd.DataFrame:
    """
    DataFrame 형태로 들어온 데이터의 "tags" column을 가지고 각 태그들을 인코딩하여
    인코딩된 태그들을 원래 DataFrame에 추가한 후 반환하는 함수입니다.

    Args:
    - df_videos (pandas.DataFrame) : 인코딩할 데이터프레임입니다.
    - creator (str) : 비디오 제작자의 이름입니다.

    Returns:
    - pandas.DataFrame : 인코딩된 데이터프레임을 반환합니다.
    """
    # 데이터프레임을 복사하여 index reset
    df_copy = df.copy().reset_index(drop=True)
    # 새로운 열 "human_count_class"와 "encoded_tags"를 NaN 값으로 추가
    df_copy["human_count_class"] = np.nan
    df_copy["encoded_tags"] = np.nan
    # encoded_human_dict 함수를 통해 태그들을 인코딩한 dictionary 생성
    list_dict = encoded_human_dict(df)

    # "encoded_tags" 컬럼의 데이터 타입을 object로 변경
    df_copy["encoded_tags"] = df_copy["encoded_tags"].astype(object)

    # 각 행의 "tags" column에서 loop 실행
    for idx, humans in enumerate(df_copy.tags.values):
        temp_list = []
        for human in humans.split(" "):
            # 각 인물에 대한 인코딩된 태그 리스트 생성
            temp_list.append(list_dict[human])
        df_copy["encoded_tags"][idx] = temp_list

    # 인물이 등장하는 횟수에 따라서 "human_count_class" 열 생성
    for idx, enc_tags in enumerate(df_copy.encoded_tags.values):
        if enc_tags[0] == list_dict[creator]:
            df_copy.human_count_class[idx] = 0
            continue
        if len(enc_tags) == 1:
            df_copy.human_count_class[idx] = 1
        else:
            df_copy.human_count_class[idx] = 2
    # "human_count_class" 열의 데이터 타입을 int로 변경
    df_copy.human_count_class = df_copy.human_count_class.astype(int)

    # 인코딩된 데이터프레임 반환
    return df_copy


def split_df_by_habang(df: pd.DataFrame) -> tuple:
    """
    주어진 DataFrame을 해시태그의 인원수 단위로 나눈 DataFrame을 반환한다.

    Args:
    - df_videos (pandas.DataFrame) : 사람 해시태그만 존재하는 데이터프레임을 인자로 받는다.

    Returns:
    - pandas.DataFrame : 혼자 / 1인과 같이 / 2인 이상과 같이 합방한 DataFrame 세 개를 tuple로 묶어 반환한다.
    """
    # 'encoded_tags' 값이 [25]인 row는 '해방군'으로 분류합니다.
    none_df = df.loc[df.encoded_tags.apply(lambda x: x == [25])]

    # 'encoded_tags' 값이 [25]가 아닌 row는 나머지 두 그룹에 속합니다.
    habang_upper1_df = df.drop(none_df.index)

    # 'habang_upper1_df'를 복사한 DataFrame을 만들어 '민족군(2차)'를 만듭니다.
    habang_upper2_df = habang_upper1_df.copy().reset_index(drop=True)

    # 'habang_upper1_df'에서 '민족군(1차)'에 속하는 row를 찾아서 temp_index에 저장합니다.
    temp_index = []
    for idx, value in enumerate(habang_upper1_df.encoded_tags.values):
        if len(value) == 1:
            temp_index += habang_upper1_df.loc[
                habang_upper1_df.encoded_tags.apply(lambda x: x == value)
            ].index.tolist()

    temp_index = list(set(temp_index))
    habang_upper1_df = habang_upper1_df.loc[temp_index]

    # 'habang_upper2_df'에서 '민족군(2차)'에 속하는 row를 찾아서 temp_index에 저장합니다.
    temp_index = []
    for idx, value in enumerate(habang_upper2_df.encoded_tags.values):
        if len(value) >= 2:
            temp_index += habang_upper2_df.loc[
                habang_upper2_df.encoded_tags.apply(lambda x: x == value)
            ].index.tolist()

    temp_index = list(set(temp_index))
    habang_upper2_df = habang_upper2_df.loc[temp_index]

    # 'habang_upper2_df'에서 'tags' column 값을 정렬하고 다시 합쳐줍니다.
    habang_upper2_df.tags = habang_upper2_df.tags.apply(lambda x: x.split(" "))
    habang_upper2_df.tags = habang_upper2_df.tags.apply(lambda x: sorted(x))
    habang_upper2_df.tags = habang_upper2_df.tags.apply(lambda x: " ".join(x))

    return none_df, habang_upper1_df, habang_upper2_df


def unique_video_id_to_dict(df: pd.DataFrame) -> dict:
    unique_video_id = df.video_id.unique()
    unique_video_id_like_count_dict = dict()

    for video_id in unique_video_id:
        unique_video_id_like_count_dict[video_id] = df.loc[
            df.video_id == video_id
        ].like_count.sum()

    return unique_video_id_like_count_dict


def video_on_like_count_sum(df):
    # 비디오별 댓글 좋아요 수
    unique_video_id_like_count_dict = unique_video_id_to_dict(df)
    # 내림차순 정렬
    comments_like_sum = sorted(
        unique_video_id_like_count_dict.items(), key=lambda x: x[1], reverse=True
    )

    video_on_comments_like_sum = list()

    for idx, value in enumerate(comments_like_sum):
        if idx == 10:
            break
        video_on_comments_like_sum.append(value)

    return video_on_comments_like_sum


def created_df(df) -> list:
    new_df = list()
    stop_words = stopwords("./데이터/stopwords.txt")
    video_on_comments_like_sum = video_on_like_count_sum(df)
    for video_id in tqdm(video_on_comments_like_sum):
        new_df.append(
            df.loc[df.video_id == video_id[0]].comment.apply(
                lambda x: text_cleaning(x, stop_words)
            )
        )
    return new_df


def word_count_stop_df(
    splited_df: pd.DataFrame, video_id: str = "", stopwords: list = []
) -> pd.DataFrame:
    """
    주어진 데이터프레임에서 특정 video_id에 해당하는 댓글들을 가져와서 불용어(stopwords)를 제외하고 반환하는 함수.

    :param splited_df: 댓글이 split 된 데이터프레임.
    :param video_id: 가져오고자 하는 댓글이 속한 동영상의 video_id.
    :param stopwords: 불용어(stopwords) 리스트.
    :return: 불용어를 제외한 댓글들이 있는 데이터프레임.
    """

    # video_id에 해당하는 댓글만 추출
    new_chim_df = splited_df.loc[splited_df.video_id == video_id].reset_index(drop=True)

    # 불용어(stopwords) 제거
    for idx, comments in enumerate(new_chim_df.comment):
        new_list = []
        for comment in comments:
            if comment not in stopwords:
                new_list.append(comment)
        new_chim_df.comment[idx] = new_list

    # 댓글이 없는 경우 제거
    for idx, comments in enumerate(new_chim_df.comment):
        if not comments:
            new_chim_df.drop(idx, inplace=True)

    return new_chim_df


def word_freq(word_count_stop_df: pd.DataFrame) -> pd.DataFrame:
    """
    주어진 데이터프레임에서 각 댓글의 단어 출현 빈도수를 계산하는 함수.

    Args:
    - word_count_stop_df : pd.DataFrame : 댓글과 같은 텍스트 데이터를 가진 데이터프레임

    Returns:
    - new_df : pd.DataFrame : 각 단어와 출현 빈도수가 기록된 새로운 데이터프레임
    """
    comm_map_second = Counter(str_list(word_count_stop_df, "comment"))

    new_df = pd.DataFrame(
        {
            "word": list(comm_map_second.keys()),
            "count": list(comm_map_second.values()),
        }
    )
    return new_df
