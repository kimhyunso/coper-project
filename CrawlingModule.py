from googleapiclient.discovery import build
import pandas as pd
import numpy as np


class YoutubeBuilder:
    __PLATFORM = 'youtube'
    __VERSION = 'v3'
    __video_id_list = list()
    __comment_list = list()
    __videos_list = list()
    __cate_gory_dict = dict()

    '''
    Attributes:
        __youtube : 발급받은 키를 통해 실제 크롤링할 수 있는 객체를 만든다(instarnce) - private
        __PLATFORM : 어디서 크롤링을 하는가 - private
        __VERSION : youtube의 API 버전 - private
        __video_id_list : 비디오아이디를 반환하기 위한 리스트 - private
        __comment_list : 댓글 내용을 반환하기 위한 리스트 - private
        __channel_id : 채널 아이디를 반환하기 위한 문자열 - private
        __videos_list : 비디오들을 반환하기 위한 리스트 - private
        __cate_gory_dict : 카테고리별 수치화된 딕셔너리 - private
        __name : 채널 명의 이름을 반환한다. - private
    ForExample:
        response = youtube.playlists().list()
        response = youtube.videos().list()
        response = youtube.channels().list()
        response = youtube.search().list()
    '''

    def __init__(self, api_key) -> None:
        """
        생성자 - constructor
        Youtbe API를 토대로 크롤링객체를 초기화(Instarnce)를 해주는 단계이다.
        https://developers.google.com/youtube/v3/getting-started
        Args:
            api_key : 발급받은 키
        """
        self.__key = api_key
        self.__youtube = build(self.__PLATFORM, self.__VERSION, developerKey=self.__key)

        # 카테고리 파일을 읽어드린다.
        inline = open("./데이터/category_id_list.txt", "r" , encoding='utf-8')
        # 파일을 읽으며 ' - ' 단위로 쪼갠다.
        for line in inline.readlines():
            key_value = line.split(" - ")
            # 카테고리 사전에서 숫자를 키로 문자를 value로 지정함
            self.__cate_gory_dict[int(key_value[0])] = key_value[1][:-1]


    def search_get_channelId(self, search_name: str, maxResults=1):
        """
        채널명을 토대로 channelId값을 알아온다.
        Args:
            search_name : 채널명을 통해서 channel_id를 검색한다.
            maxResults : 한 번에 받을 수 있는 결과값
        """
        # youtube searchAPI를 통해 channel_id를 가져옴
        self.__response = (
            self.__youtube.search()
            .list(
                part="id,snippet", q=search_name, type="channel", maxResults=maxResults
            )
            .execute()
        )
        # resopnse item 안에 id 안에 channelId가 있음
        return self.__response["items"][0]['id']['channelId']
    
    def channel_get_view_count(self, channelId:str, maxResults=1) -> int:
        self.__response = self.__youtube.channels().list(part='id,snippet,statistics', id=channelId, maxResults=1).execute()
        return self.__response['items']['statistics']['viewCount']

    def search_get_videoId_in_channel(self, channelId:str, maxResults=50) -> list:
        '''
        channel_id를 토대로 video_id들을 알아온다.
        Args:
            channelId : 알아오고 싶은 채널의 ID
            maxResults : 한 번에 받을 수 있는 결과값
        '''
        # youtube searchAPI를 통해 video_id들을 가져옴
        self.__response = (
            self.__youtube.search()
            .list(
                part="id,snippet",
                channelId=channelId,
                type="video",
                maxResults=maxResults,
                order="date",
            )
            .execute()
        )

        # 최대 몇개 까지 제한을 할 것인가
        count = 0

        # response 안에서 데이터들을 추출해옴
        while self.__response:

            # response 안의 items안에 id안에 videoId가 있음
            for item in self.__response["items"]:
                self.__video_id_list.append(item["id"]["videoId"])
            # count를 돌면서 100이 되면 while문을 빠져나옴 maxResults에 따라 달라짐 ex) maxResults 50개 -> 50 * 100 => 5000개의 데이터를 가져옴
            if count == 20:
                break

            # 다음 데이터가 있을 경우 다시 요청을 함
            if "nextPageToken" in self.__response:
                self.__response = (
                    self.__youtube.search()
                    .list(
                        part="id,snippet",
                        channelId=channelId,
                        type="video",
                        pageToken=self.__response["nextPageToken"],
                        maxResults=maxResults,
                        order="date",
                    )
                    .execute()
                )
            # 요청을 한 번 할때 break를 할 수 있도록 +를 해줌
            count += 1
        return self.__video_id_list

    def get_videos_in_videoId_list(
        self, videoId_list: list, regionCode="kr", maxResults=50
    ) -> list:
        """
        https://developers.google.com/youtube/v3/docs/videos/list
        Args:
            videoId_list : video_id 목록을 받는다.
            regionCode : ISO 3166-1 alpha-2 국가 코드값
            maxResults : 한 번에 받을 수 있는 결과값
        """
        # 최대 몇개 까지 제한을 할 것인가
        count = 0
        # 비디오 id의 갯수 만큼 돈다.
        for videoId in videoId_list:
            # youtube videos API를 통해 요청을 한다.
            self.__response = (
                self.__youtube.videos()
                .list(
                    part="snippet,id,statistics",
                    id=videoId,
                    regionCode=regionCode,
                    maxResults=maxResults,
                )
                .execute()
            )

            # response의 items안에 snippet 안에 categoryId를 가져온다.
            while self.__response:
                for item in self.__response["items"]:
                    # category_dict 사전에 등록된 카테고리 이름을 가져온다
                    category_name = self.__cate_gory_dict[
                        int(item["snippet"]["categoryId"])
                    ]
                    self.__videos_list.append(
                        [
                            videoId,
                            int(item["snippet"]["categoryId"]),
                            category_name,
                            item["snippet"]["title"],
                            int(item["statistics"]["viewCount"]),
                            int(item["statistics"]["likeCount"]),
                            item["snippet"]["publishedAt"],
                            item["snippet"]["description"],
                        ]
                    )
                    # count를 돌면서 100이 되면 while문을 빠져나옴 maxResults에 따라 달라짐 ex) maxResults 50개 -> 50 * 100 => 5000개의 데이터를 가져옴
                    if count == 20:
                        break

                    # 다음 데이터가 있을 경우 다시 요청함
                    if "nextPageToken" in self.__response:
                        self.__response = (
                            self.__youtube.videos()
                            .list(
                                part="snippet,id,statistics",
                                id=videoId,
                                regionCode=regionCode,
                                maxResults=maxResults,
                            )
                            .execute()
                        )
                    # 요청을 한 번 할때 break를 할 수 있도록 +를 해줌
                    count += 1
                break
        return self.__videos_list

    def get_comments(self, video_id_list, maxResults=50) -> list:

        """
        YoutubeAPI를 통해 동영상의 댓글들을 가져온다.
        https://developers.google.com/youtube/v3/docs/commentThreads/list
        Args:
            video_id_list : 동영상의 ID 목록 ex) https://www.youtube.com/watch?v=BP1rFQtacU4 :: BP1rFQtacU4
            maxResults : 한 번에 몇개의 결과를 받을 것인가
        """
        
        # 비디오 개수 만큼 돌며 youtube에게 요청을 한다.
        try:
            for video_id in video_id_list:
                self.__response = (
                    self.__youtube.commentThreads()
                    .list(
                        part="id,snippet,replies", videoId=video_id, maxResults=maxResults
                    )
                    .execute()
                )
                # response의 개수 만큼 돌며 댓글들을 가져온다.
                while self.__response:
                    for item in self.__response["items"]:
                        comment = item["snippet"]["topLevelComment"]["snippet"]
                        self.__comment_list.append(
                            [
                                comment["videoId"],
                                item["snippet"]["topLevelComment"]["id"],
                                comment["textOriginal"],
                                int(comment["likeCount"]),
                                comment["publishedAt"],
                                comment["updatedAt"],
                            ]
                        )

                    # 다음 댓글이 있다면 다시 재요청을 보낸다.
                    if "nextPageToken" in self.__response:
                        self.__response = (
                            self.__youtube.commentThreads()
                            .list(
                                part="id,snippet,replies",
                                videoId=video_id,
                                pageToken=self.__response["nextPageToken"],
                                maxResults=maxResults,
                            )
                            .execute()
                        )
                    else:
                        break
        except:
            pass    
        return self.__comment_list
