#### Check the quotas for today before use it

from googleapiclient.discovery import build
import config
from datetime import datetime
import json
import time
import os
import pandas as pd
import re
from json import JSONDecoder, JSONDecodeError

KIDS = config.KIDS
class LinkCollector(object):

    def __init__(self, query_key_word):
        self.query_key_word = query_key_word
        self.resource = build('youtube', 'v3', developerKey=KIDS)
        self.cur_res = None
        self.res = []

    def search(self, next_page_token=None):

        cur_page_token = next_page_token
        import copy
        req = self.resource.search().list(part='snippet',
                                          maxResults=50,
                                          order='relevance',
                                          pageToken=next_page_token,
                                          q=self.query_key_word,
                                          relevanceLanguage='en',
                                          type='video',
                                          videoCaption='any'
                                          )
        res = req.execute()
        self.cur_res = res
        self.res.append(copy.deepcopy(self.cur_res))
        if res is not None:
            self.__persist_data(res, cur_page_token)

    def __generate_datetime(self, year, month, day):
        return datetime(year=year, month=month, day=day).strftime('%Y-%m-%dT%H:%M:%SZ')

    def __persist_data(self, res, cur_page_token):
        if cur_page_token is None:
            cur_page_token = ''
        if not os.path.exists('collected_link_json'):
            os.mkdir('collected_link_json')
        json.dump(res, open('collected_link_json/youtube_request_' + cur_page_token + '.json', 'a+'))

def extract_video_from_json(directory):

    videos = []
    for entry in os.scandir(directory):
        if entry.path.endswith(".json") and entry.is_file():
            items_ls = []
            try:
                items_ls.extend([json.load(open(entry, 'r'))])
            except:
                for items_piece in stream_json(open(entry, 'r')):
                    items_ls.append(items_piece)

            new_videos = [item['id']['videoId'] for items in items_ls for item in items['items']]
            videos.extend(new_videos)

    return videos

class VideoDetailsCollector(object):

    def __init__(self, video_list):
        self.video_list = video_list
        self.items = []
        self.resource = build('youtube', 'v3', developerKey=KIDS)

    def retrieve(self):

        for i in range(0, len(self.video_list), 50):
            query_video = ','.join(self.video_list[i:i+50])
            req = self.resource.videos().list(part = 'snippet,contentDetails,statistics,topicDetails',
                                              maxResults = 50,
                                              id = query_video,
                                              )

            import copy
            res = req.execute()
            self.items.append(copy.deepcopy(res))
            if res is not None:
                self.__persist_data(res, self.video_list[i] + '_' + str(i))
            time.sleep(2)

    def __generate_datetime(self, year, month, day):
        return datetime(year=year, month=month, day=day).strftime('%Y-%m-%dT%H:%M:%SZ')

    def __persist_data(self, res, unique_identifier):
        if unique_identifier is None:
            unique_identifier = ''
        if not os.path.exists('collected_video_detail'):
            os.mkdir('collected_video_detail')
        json.dump(res, open('collected_video_detail/youtube_request_' + unique_identifier + '.json', 'a+'))

def extract_video_tag_des(directory):

    videos_df = pd.DataFrame(columns = ['video', 'title', 'tags', 'description'])

    for entry in os.scandir(directory):
        if entry.path.endswith(".json") and entry.is_file():
            items = json.load(open(entry, 'r'))
            video_ids = [item['id'] for item in items['items']]
            video_titles = [item['snippet']['title'] for item in items['items']]
            video_tags = [','.join(item['snippet']['tags']) if 'tags' in item['snippet'] else [] for item in items['items']]
            video_desc = [item['snippet']['description'] for item in items['items']]
            videos_df = pd.concat([videos_df, pd.DataFrame({'video': video_ids,
                                                     'title': video_titles,
                                                     'tags': video_tags,
                                                     'description': video_desc})])

    return videos_df

def stream_json(file_obj, buf_size=1024, decoder=JSONDecoder()):

    ### deal with multiple json object in on file

    NOT_WHITESPACE = re.compile(r"[^\s]")

    buf = ""
    ex = None
    while True:
        block = file_obj.read(buf_size)
        if not block:
            break
        buf += block
        pos = 0
        while True:
            match = NOT_WHITESPACE.search(buf, pos)
            if not match:
                break
            pos = match.start()
            try:
                obj, pos = decoder.raw_decode(buf, pos)
            except JSONDecodeError as e:
                ex = e
                break
            else:
                ex = None
                yield obj
        buf = buf[pos:]
    if ex is not None:
        raise ex


if __name__ == '__main__':

    link_collector = LinkCollector('Kids toys')
    count = 0
    next_page_token = None
    while count < 30 and (count==0 or next_page_token is not None):

        link_collector.search(next_page_token)
        print(next_page_token)
        if (len(link_collector.res[-1]['items']) == 0) or ('nextPageToken' not in link_collector.cur_res):
            break
        next_page_token = link_collector.cur_res['nextPageToken']
        time.sleep(5)
        count += 1

    video_list = extract_video_from_json('collected_link_json')

    video_detail_collector = VideoDetailsCollector(video_list = video_list)
    video_detail_collector.retrieve()
    res_df = extract_video_tag_des('collected_video_detail')
    res_df.to_csv('toys_video_details.csv')





