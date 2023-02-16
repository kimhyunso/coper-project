from googleapiclient.discovery import build

class YoutubeBulider:
    def __init__(self, api_key):
        '''
        Youtbe API를 토대로 크롤링객체를 초기화(Instarnce)를 해주는 단계이다.
        https://developers.google.com/youtube/v3/getting-started
        Args:
            api_key : 발급받은 키
        Attributes:
            __key : 발급받은 키 - private
            __youtube : 발급받은 키를 통해 실제 크롤링할 수 있는 객체를 만든다(instarnce) - private
            __PLATFORM : 어디서 크롤링을 하는가 - private
            __VERSION : youtube의 API 버전
        Description:
            response = youtube.playlists().list()
            response = youtube.videos().list()
            response = youtube.channels().list()
            response = youtube.search().list()
        '''
        __PLATFORM = 'youtube'
        __VERSION = 'v3'

        self.__key = api_key
        self.__youtube = build(__PLATFORM, __VERSION, developerKey=self.get_key)

    def get_comment_threads(self, *part, videoId, maxResults=20) -> list:
        '''
        > return list
        https://developers.google.com/youtube/v3/docs/commentThreads/list
        Args:
            *part : id, replies, snippet - 필수매개변수
            videoId : 동영상의 ID ex) https://www.youtube.com/watch?v=BP1rFQtacU4 :: BP1rFQtacU4
        Attributes:
            response : commentThreads에서 목록의 쿼리를 날려서 결과를 도출
        '''
        self.__response = self.get_youtube().commentThreads().list(part=f'{part[0]},{part[1]}', videoId=videoId, maxResults=maxResults).execute()
        return self.get_response()
    

    def get_videos(self, *part, chart='mostPopular', regionCode='kr') -> list:
        '''
        > return list
        https://developers.google.com/youtube/v3/docs/videos/list
        Args:
            **part : id, snippet, contentDetails, fileDetails, liveStreamingDetails, player, processingDetails, recordingDetails, statistics, status, suggestions, topicDetails - 필수매개변수
            chart : mostPopular - 인기동영상
            regionCode : 이 매개변수를 사용하는 경우 차트 또한 설정해야 합니다. 이 매개변수 값은 ISO 3166-1 alpha-2 국가 코드
        Attributes:
            response : Videos에서 목록의 쿼리를 날려서 결과를 도출
        '''

        # self.__response = self.get_youtube().videos().list(part=f'{part[0]},{part[1]}', chart=chart, regionCode=regionCode).execute()
        print(self.get_youtube())
        return self.get_response()

    def get_key(self):
        return self.__key
    def set_key(self, value):
        self.__key = value
    def get_youtube(self):
        self.__youtube
    def set_youtube(self, value):
        self.__youtube = value

    def get_response(self):
        self.__response

    
    




        
 
            

