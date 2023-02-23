from googleapiclient.discovery import build
import pandas as pd
import numpy as np

class YoutubeBulider():
    __PLATFORM = 'youtube'
    __VERSION = 'v3'
    __video_id_list = list()
    __comment_list = list()
    __category_list = list()
    __channel_id = ''
    __views_count_list = list()

    def __init__(self, api_key) -> None:
        '''
        Youtbe API를 토대로 크롤링객체를 초기화(Instarnce)를 해주는 단계이다.
        https://developers.google.com/youtube/v3/getting-started
        Args:
            api_key : 발급받은 키
        Attributes:
            __key : 발급받은 키 - private
            __youtube : 발급받은 키를 통해 실제 크롤링할 수 있는 객체를 만든다(instarnce) - private
            __PLATFORM : 어디서 크롤링을 하는가 - private
            __VERSION : youtube의 API 버전 - private
            __video_id_list : 인기 급상승 동영상의 video_id를 반환하기 위한 list - private
            __comment_list : 인기 급상승 동영상의 댓글들을 반환하기 위한 list - private
        Description:
            response = youtube.playlists().list()
            response = youtube.videos().list()
            response = youtube.channels().list()
            response = youtube.search().list()
        FOREX:
            UCZ0bi2aVJngKLwFTU5g_fLQ
            https://www.googleapis.com/youtube/v3/videos?part=statistics,snippet,id&id=ljP6X7gfNu8&regionCode=KR
            https://www.googleapis.com/youtube/v3/commentThreads?part=replies,snippet&allThreadsRelatedToChannelId=UCZ0bi2aVJngKLwFTU5g_fLQ
            https://www.googleapis.com/youtube/v3/search?part=snippet&maxResults=25&channelId=UCQ2O-iftmnlfrBuNsUUTofQ&type=video&key=AIzaSyCAH9UmLLAcjgLqMVwJ4VgK32qlNX6F6z8
        '''
        self.__key = api_key
        self.__youtube = build(self.__PLATFORM, self.__VERSION, developerKey=self.__key)


    def get_videoId_in_channel(self, channelId:str, maxResults=50) -> list:
        self.__response = self.__youtube.search().list(part='id,snippet', channelId=channelId, type='video', maxResults=maxResults, order='date').execute()
        
        count = 0 

        while self.__response:
            for item in self.__response['items']:
                self.__video_id_list.append(item['id']['videoId'])

            if count == 1:
                break
            if 'nextPageToken' in self.__response:
                self.__response = self.__youtube.search().list(part='id,snippet', channelId=channelId, type='video', pageToken=self.__response['nextPageToken'], maxResults=maxResults, order='date').execute()
            count += 1

        return self.__video_id_list


    def search_channelId(self, search_name:str, maxResults=1) -> str:
        self.__response = self.__youtube.search().list(part='id,snippet', q=search_name, type='channel', maxResults=maxResults).execute()
        for item in self.__response['items']:
            self.__channel_id = item['id']['channelId']
        return self.__channel_id

    def get_categoryId_in_channel(self, videoId_list:list, regionCode='kr', maxResults=1) -> list:
        '''
        > return list
        YoutubeAPI를 통해 인기급상승 영상ID(video_id)를 크롤링해온다.
        https://developers.google.com/youtube/v3/docs/videos/list
        Args:
            **part : id, snippet, contentDetails, fileDetails, liveStreamingDetails, player, processingDetails, recordingDetails, statistics, status, suggestions, topicDetails - 필수매개변수
            chart : mostPopular - 인기동영상
            regionCode : 이 매개변수를 사용하는 경우 차트 또한 설정해야 합니다. 이 매개변수 값은 ISO 3166-1 alpha-2 국가 코드
            maxResults : 몇 개의 결과를 반환받을 것인지
        Attributes:
            response : Videos에서 목록의 쿼리를 날려서 결과를 도출 - private
        '''
        # 백번
        for videoId in videoId_list:
            # 백번 쿼리 날림
            
            self.__response = self.__youtube.videos().list(part='snippet,id,statistics', id=videoId, regionCode=regionCode, maxResults=maxResults).execute()

            while self.__response:
                for item in self.__response['items']:
                    self.__category_list.append(item['snippet']['categoryId'])
                    self.__views_count_list.append([videoId, item['statistics']['viewCount'], item['statistics']['likeCount']])
                break
        
        return self.__category_list, self.__views_count_list

        

    def get_comments(self, video_id_list, maxResults=100) -> list:
        '''
        > return list
        YoutubeAPI를 통해 인기급상승의 댓글들을 크롤링해온다.
        https://developers.google.com/youtube/v3/docs/commentThreads/list
        Args:
            video_id_list : 동영상의 ID 목록 ex) https://www.youtube.com/watch?v=BP1rFQtacU4 :: BP1rFQtacU4
            maxResults : 몇 개의 결과를 반환받을 것인지
        Attributes:
            response : commentThreads에서 목록의 쿼리를 날려서 결과를 도출 - private
        '''

        for video_id in video_id_list:
            self.__response = self.__youtube.commentThreads().list(part='snippet,replies', videoId=video_id, maxResults=maxResults).execute()
            while self.__response:
                for item in self.__response['items']:
                    comment = item['snippet']['topLevelComment']['snippet']
                    self.__comment_list.append([comment['textOriginal'], comment['authorDisplayName'], comment['publishedAt'], comment['updatedAt'], comment['likeCount'], video_id])
                if 'nextPageToken' in self.__response:
                    self.__response = self.__youtube.commentThreads().list(part='snippet,replies', videoId=video_id, pageToken=self.__response['nextPageToken'], maxResults=maxResults).execute()
                else:
                    break
        return self.__comment_list


    def create_df(self, comment_list) -> list:
        '''
        Args:
            comment_list : 인기 동영상의 댓글 목록
        Attributes:
            comments_df : pd.
        '''
        comments_df = pd.DataFrame(comment_list)
        return comments_df




        
 
            

