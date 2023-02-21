#%%
import pandas as pd
import numpy as np
import Preprocessing as prep
import Modeling as mo

from konlpy.tag import Okt


file_path_stopwords = "C:/Users/TECH2_07/Desktop/이도원 프로젝트 폴더/자료/1차시도/stopwords.txt"
stopwords_list = ['제주도', '호스텔', '위치', '대중교통', '타고', '어디', '공항', '잠깐', '직원', '택시', '상품',]
stopwords_list = prep.stopwords(file_path_stopwords, stopwords_list)


file_path = "C:/Users/TECH2_07/Desktop/이도원 프로젝트 폴더/자료/1차시도/changed_label.csv"
new_df = pd.read_csv(file_path, encoding='utf-8', engine='python')


import Preprocessing as prep

prep.invert_text_to_vect(prep.text_cleaning(new_df.loc[1, 'text'], stopwords_list))


#%%
prep.text_cleaning(new_df.loc[1, 'text'], stopwords_list)
# prep.text_cleaning(new_df.loc[1, 'text'], stopwords_list)










# %%
