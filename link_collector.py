#### Check the quotas for today before use it

from googleapiclient.discovery import build
import config
from datetime import datetime
import json
import time
import os
import pandas as pd
import re
import argparse
from json import JSONDecoder, JSONDecodeError
import io
import nltk
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('words')
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.corpus import words
stop_words = set(stopwords.words('english'))
stopword_set = set(stopwords.words())
word_set = set(words.words())

from sklearn.feature_extraction.text import TfidfVectorizer
import seaborn as sns
import matplotlib.pyplot as plt
plt.switch_backend('Agg')
### set up plotting parameters
from matplotlib import rcParams
plt.style.use('seaborn-poster')
plt.rcParams['font.family'] = 'serif'
rcParams['font.sans-serif'] = ['Palatino']
rcParams['figure.max_open_warning'] = 30

KIDS = config.KIDS
class LinkCollector(object):

    def __init__(self, query_key_word, folder_name_key_words):
        self.query_key_word = query_key_word
        self.folder_name_key_word = folder_name_key_words
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
                                          videoCaption= 'closedCaption' # ''closedCaption' #'any', 'none'
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
        persist_folder = 'collected_link_json_' + self.folder_name_key_word
        if not os.path.exists(persist_folder):
            os.mkdir(persist_folder)
        json.dump(res, open(os.path.join(persist_folder, 'youtube_request_' + cur_page_token + '.json'), 'a+'))

    def get_persisted_data_dir(self):
        return 'collected_link_json_' + self.folder_name_key_word


class VideoDetailsCollector(object):

    def __init__(self, video_list, dir_name_key_word = None):
        self.video_list = video_list
        self.items = []
        self.resource = build('youtube', 'v3', developerKey=KIDS)
        self._dir_name_key_word = dir_name_key_word

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

    @property
    def dir_name_key_word(self):
        return self._dir_name_key_word

    @dir_name_key_word.setter
    def dir_name_key_word(self, name):
        if self.dir_name_key_word is not None:
            self._dir_name_key_word = name
        else:
            return

    def __generate_datetime(self, year, month, day):
        return datetime(year=year, month=month, day=day).strftime('%Y-%m-%dT%H:%M:%SZ')

    def __persist_data(self, res, unique_identifier):
        if unique_identifier is None:
            unique_identifier = ''
        persisted_data_dir = 'collected_video_detail_' + self.dir_name_key_word
        if not os.path.exists(persisted_data_dir):
            os.mkdir(persisted_data_dir)
        json.dump(res, open(os.path.join(persisted_data_dir, 'youtube_request_' + unique_identifier + '.json'), 'a+'))

    def get_persisted_data_directory(self):
        return 'collected_video_detail_' + self.dir_name_key_word

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

def extract_video_tag_des(directory, category = None):

    videos_df = pd.DataFrame(columns = ['video', 'title', 'tags', 'description', 'category', 'caption'])

    for entry in os.scandir(directory):
        if entry.path.endswith(".json") and entry.is_file():
            items_ls = []
            try:
                items_ls.extend([json.load(open(entry, 'r'))])
            except:
                for items_piece in stream_json(open(entry, 'r')):
                    items_ls.append(items_piece)

            video_ids = [item['id'] for items in items_ls for item in items['items']]
            video_titles = [item['snippet']['title'] for items in items_ls for item in items['items']]
            video_tags = [','.join(item['snippet']['tags']) if 'tags' in item['snippet'] else []
                          for items in items_ls for item in items['items']]
            video_desc = [item['snippet']['description'] for items in items_ls for item in items['items']]
            video_caption = [item['contentDetails']['caption'] for items in items_ls for item in items['items']]
            videos_df = pd.concat([videos_df, pd.DataFrame({'video': video_ids,
                                                     'title': video_titles,
                                                     'tags': video_tags,
                                                     'description': video_desc,
                                                     'category': category,
                                                     'caption': video_caption})])

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

def remove_stop(text):
    try:
        text_tokens = word_tokenize(text)
        tokens_without_sw = [word for word in text_tokens
                             if (word.lower().strip() not in stopword_set) and (word.lower().strip() in word_set)]
        return tokens_without_sw
    except:
        return ['']


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--search_word')
    parser.add_argument('--folder_name_key_words')
    parser.add_argument('--csv_result_dir')
    args = parser.parse_args()

    search_word = args.search_word
    folder_name_key_words = args.folder_name_key_words
    csv_result_dir = args.csv_result_dir
    if not os.path.exists(csv_result_dir):
        os.mkdir(csv_result_dir)
    link_collector = LinkCollector(search_word, folder_name_key_words)  ### use to collect related link
    count = 0
    next_page_token = None
    while count < 50 and (count==0 or next_page_token is not None):
        ### make sure it won't use all quotes from youtubeapi (10000/perday, 1 per video query, 100 per key word search)

        link_collector.search(next_page_token)
        print(next_page_token)
        if (len(link_collector.res[-1]['items']) == 0) or ('nextPageToken' not in link_collector.cur_res):
            break
        next_page_token = link_collector.cur_res['nextPageToken']
        time.sleep(5)
        count += 1

    print('collected {0!r} times'.format(count))
    link_collector_dir = link_collector.get_persisted_data_dir()
    print(link_collector_dir)
    video_list = extract_video_from_json(link_collector_dir)

    video_detail_collector = VideoDetailsCollector(video_list = video_list, dir_name_key_word=folder_name_key_words)
    video_detail_collector.retrieve()
    video_detail_dir = video_detail_collector.get_persisted_data_directory()
    res_df = extract_video_tag_des(video_detail_dir, category=search_word)
    res_df = res_df.drop_duplicates(subset=['video', 'title'])
    print(len(res_df))
    res_df.to_csv(os.path.join(csv_result_dir, 'video_details_' + folder_name_key_words +'.csv'))

    # ### speed up searching with set
    #
    # res_df['tags_w/o_stop'] = res_df.tags.apply(remove_stop)
    # res_df['desc_w/o_stop'] = res_df.description.apply(remove_stop)
    #
    # tags_corpus = [','.join(x) for x in res_df['tags_w/o_stop']]
    # desc_corpus = [','.join(x) for x in res_df['desc_w/o_stop']]
    # tags_vectorizer = TfidfVectorizer()
    # tags_X = tags_vectorizer.fit_transform(tags_corpus)
    # desc_vectorizer = TfidfVectorizer()
    # desc_X = desc_vectorizer.fit_transform(desc_corpus)
    # new_tags_x = pd.DataFrame(tags_X.toarray(),
    #                           columns=tags_vectorizer.get_feature_names(), index=res_df['video'])
    #
    # new_desc_x = pd.DataFrame(desc_X.toarray(),
    #                           columns=desc_vectorizer.get_feature_names(), index=res_df['video'])
    #
    # first_30 = new_tags_x.sum().sort_values(ascending=False)[
    #            3:30]  ### the first 4 are either toy or kids (search terms)
    # frist_30_in_desc = new_desc_x.sum().sort_values(ascending=False)[:30]
    #
    # plt.figure(figsize=(20, 8))
    # sns.barplot(x=first_30.index, y=first_30)
    # plt.xticks(rotation=60, fontsize=12)
    # plt.xlabel('Key Words')
    # plt.ylabel('Accumulated TFIDF')
    # plt.savefig('Tags_tfidf.pdf')
    #
    # plt.figure(figsize=(20, 8))
    # sns.barplot(x=frist_30_in_desc.index, y=frist_30_in_desc)
    # plt.xticks(rotation=60, fontsize=12)
    # plt.xlabel('Description Key Words')
    # plt.ylabel('Accumulated TFIDF')
    # plt.savefig('description_tfidf.pdf')





