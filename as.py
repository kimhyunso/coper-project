# YoutubeModule Class
#from ./CrawlingModule import YoutubeBuilder

# KEY 파일로 읽어오기
# KEY_FILE = open('./API_KEY.txt', 'r')

# # API_KEY 할당
# API_KEY = KEY_FILE.readline().split(':')[1] 

# # YoutbeModule Instarnce 
# bulider = YoutubeBuilder(API_KEY)
# channelId = bulider.search_get_channelId('침착맨')
# video_id_list = bulider.search_get_videoId_in_channel(channelId=channelId)
# video_list = bulider.get_videos_in_videoId_list(videoId_list=video_id_list)
# comments_list = bulider.get_comments(video_id_list)
# import pandas as pd

# # video_id,category_id,category_name,title,views_count,like_count,uploaded_at
# chim_df = pd.DataFrame(video_list, columns=['video_id', 'category_id', 'category_name', 'title', 'views_count', 'like_count', 'uploaded_at', 'tags'])

# # chim_df.created_at = pd.to_datetime(chim_df.uploaded_at)
# # df.to_csv('./데이터/침착맨_videos.csv', index=False)

# chim_df
# import pandas as pd
# import Preprocessing as prep
# chim_df = pd.read_csv('./데이터/침착맨_videos.csv', encoding='utf-8', engine='python')
# chim_df.uploaded_at = pd.to_datetime(chim_df.uploaded_at)


# chim_df.sort_values(by='views_count', ascending=False).head(10)
# chim_df.sort_values(by='like_count', ascending=False).head(10)

# chim_df.description = chim_df.description.apply(lambda x : prep.extract_hashtags(x, name='#침착맨 '))
# chim_df.description = chim_df.description.apply(lambda x : prep.remove_other_hashtag(x, chim_df, './chim_dict.txt'))
import pandas as pd
import numpy as np
import Preprocessing as prep
from konlpy.tag import Twitter
from wordcloud import WordCloud
from matplotlib import pyplot as plt
from PIL import Image
from collections import Counter

chim_comment_df = pd.read_csv('./데이터/chim_comments.csv', encoding='utf-8', engine='python')
chim_comment_df
print(chim_comment_df)
video_comment_list = list()
stopwords = prep.stopwords('./데이터/stopwords.txt')
unique_video_id = chim_comment_df.video_id.unique()

video_id_and_count = dict()

for video_id in unique_video_id:
    video_id_and_count[video_id] = chim_comment_df.loc[chim_comment_df.video_id == video_id].like_count.sum()
# 비디오별 댓글 좋아요 수
comments_like_sum = sorted(video_id_and_count.items(), key=lambda x:x[1], reverse=True)

video_on_comments_like_sum = list()

for idx, value in enumerate(comments_like_sum):
    if idx == 10:
        break
    video_on_comments_like_sum.append(value)

video_on_comments_like_sum
twitter = Twitter()
test_list = []
new_df = list()

for video_id in video_on_comments_like_sum:
    new_df.append(pd.DataFrame(chim_comment_df.loc[chim_comment_df.video_id == video_id[0]].comment.apply(twitter.nouns)))
    

count = 0
text_list = list()
str = ''
font_path = r'C:/Windows/Fonts/malgun.ttf'
youtube_mask = np.array(Image.open('./데이터/youtube_logo.jfif'))

for df in new_df:
    for i in df.comment.iloc[range(len(df))]:
        str += ' '.join(i)
    text_list.append(str)
    print(text_list)
    print(df)
text_list
from wordcloud import WordCloud
from matplotlib import pyplot as plt

from PIL import Image


font_path = r'C:/Windows/Fonts/malgun.ttf'

youtube_mask = np.array(Image.open('./yo.jfif'))

wc = WordCloud(font_path=font_path, background_color='white', stopwords=['개소리', '존나', '개'], mask=youtube_mask, max_font_size=30, scale=7).generate(text_list[0])
plt.figure(figsize=(10, 5))
plt.axis('off')
plt.imshow(wc, interpolation='bilinear')
plt.show()
# wc.to_file(filename="침착맨_복숭아_대전.png")

