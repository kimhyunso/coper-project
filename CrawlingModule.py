from googleapiclient.discovery import build
import pandas as pd
import numpy as np


class YoutubeBulider:
    __PLATFORM = 'youtube'
    __VERSION = 'v3'
    __video_id_list = list()
    __comment_list = list()
    __channel_id = ''
    __videos_list = list()
    __cate_gory_dict = dict()
    __name = ''

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
        inline = open("./category_id_list.txt", "r" , encoding='utf-8')
        # 파일을 읽으며 ' - ' 단위로 쪼갠다.
        for line in inline.readlines():
            key_value = line.split(" - ")
            # dict
            self.__cate_gory_dict[int(key_value[0])] = key_value[1][:-1]


    

    def search_get_channelId(self, search_name: str, maxResults=1) -> str:
        """
        채널명을 토대로 channelId값을 알아온다.
        Args:
        """
        self.__response = (
            self.__youtube.search()
            .list(
                part="id,snippet", q=search_name, type="channel", maxResults=maxResults
            )
            .execute()
        )

        for item in self.__response["items"]:
            self.__channel_id = item["id"]["channelId"]
        return self.__channel_id


    def search_get_videoId_in_channel(self, channelId: str, maxResults=50) -> list:
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
        count = 1

        while self.__response:
            for item in self.__response["items"]:
                self.__video_id_list.append(item["id"]["videoId"])
            if count == 100:
                break

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
                count += 1
        return self.__video_id_list

    def get_videos_in_videoId_list(
        self, videoId_list: list, regionCode="kr", maxResults=50
    ) -> list:
        """
        > return list
        https://developers.google.com/youtube/v3/docs/videos/list
        Args:
            videoId_list : video_id를 받는다.
            regionCode : 이 매개변수를 사용하는 경우 차트 또한 설정해야 합니다. 이 매개변수 값은 ISO 3166-1 alpha-2 국가 코드
            maxResults : 몇 개의 결과를 반환받을 것인지
        Attributes:
            response : Videos에서 목록의 쿼리를 날려서 결과를 도출 - private
        """
        count = 0
        for videoId in videoId_list:
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

            while self.__response:
                for item in self.__response["items"]:
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
                    if count == 100:
                        break

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
                    count += 1
                break
        return self.__videos_list

    def get_comments(self, video_id_list, maxResults=100) -> list:
        """
        > return list
        YoutubeAPI를 통해 인기급상승의 댓글들을 크롤링해온다.
        https://developers.google.com/youtube/v3/docs/commentThreads/list
        Args:
            video_id_list : 동영상의 ID 목록 ex) https://www.youtube.com/watch?v=BP1rFQtacU4 :: BP1rFQtacU4
            maxResults : 몇 개의 결과를 반환받을 것인지
        Attributes:
            response : commentThreads에서 목록의 쿼리를 날려서 결과를 도출 - private
        """

        for video_id in video_id_list:
            self.__response = (
                self.__youtube.commentThreads()
                .list(
                    part="id,snippet,replies", videoId=video_id, maxResults=maxResults
                )
                .execute()
            )

            while self.__response:
                for item in self.__response["items"]:
                    comment = item["snippet"]["topLevelComment"]["snippet"]
                    self.__comment_list.append(
                        [
                            comment["videoId"],
                            item["snippet"]["topLevelComment"]["id"],
                            comment["textOriginal"],
                            comment["likeCount"],
                            comment["publishedAt"],
                            comment["updatedAt"],
                        ]
                    )
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
        
        return self.__comment_list
