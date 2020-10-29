from youtube_transcript_api import YouTubeTranscriptApi
import config
import argparse
import pandas as pd
import pdb
import time
import requests
import pickle
import os

def persist_transcript(transcripts):

    transcripts_dict, unfound = transcripts
    pd.DataFrame(unfound).to_csv('./unfound.csv', mode='a+', index=False, header=False)
    if not os.path.exists('./video_caption_pickle'):
        os.mkdir('./video_caption_pickle')
    for key in transcripts_dict.keys():
        pickle.dump(transcripts_dict[key], open(os.path.join('./video_caption_pickle', key+'.p'), 'wb+'))

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--video_details_file')

    args = parser.parse_args()
    video_details_file = args.video_details_file

    proxy_host = "proxy.crawlera.com"
    proxy_port = "8010"
    proxy_auth = config.proxy_auth
    proxies = {"https": "https://{}@{}:{}/".format(proxy_auth, proxy_host, proxy_port),
               "http": "http://{}@{}:{}/".format(proxy_auth, proxy_host, proxy_port)}

    video_detail_df = pd.read_csv(video_details_file, index_col = 0)
    video_ids = list(video_detail_df[video_detail_df['caption'] == True]['video'])

    video_ids = ['-TIkkGSHWeM']
    for i in range(0, len(video_ids), 50):
        print('start to extract videos {0!r}'.format(str(i)))
        video_ids_sub = video_ids[i:i+50]
        transcripts = YouTubeTranscriptApi.get_transcripts(video_ids_sub, languages=['en'],
                                                           continue_after_error= True, proxies=proxies)
        persist_transcript(transcripts)
        print(str(i))
        # time.sleep(600)

