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

# barplot 시각화
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