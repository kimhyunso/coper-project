from googleapiclient.discovery import build

class YoutubeComment:
    def __init__(self, videoId, maxResults):
        text_file = open('./API_KEY.txt', 'r')
        api_key = text_file.readline()  
        self.key = api_key
        self.videoId = videoId
        self.comments = list()
        self.maxResults = maxResults
        self.api_obj = build('youtube', 'v3', developerKey=self.key)
        response = self.api_obj.commentThreads().list(part='snippet,replies', videoId=self.videoId, maxResults=self.maxResults).execute()
        self.response = response

    def get_comments(self):
        while self.response:
            for item in self.response['items']:
                comment = item['snippet']['topLevelComment']['snippet']
                self.comments.append([comment['textOriginal'], comment['authorDisplayName'], comment['publishedAt'], comment['updatedAt'], comment['likeCount']])
                if item['snippet']['totalReplyCount'] > 0:
                    for reply_item in item['replies']['comments']:
                        reply = reply_item['snippet']
                        self.comments.append([reply['textOriginal'], reply['authorDisplayName'], reply['publishedAt'], reply['updatedAt'], reply['likeCount']])
        
            if 'nextPageToken' in response:
                response = self.api_obj.commentThreads().list(part='snippet,replies', videoId=self.videoId, pageToken=response['nextPageToken'], maxResults=self.maxResults).execute()
            else:
                break
            

