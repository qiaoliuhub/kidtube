from youtube_transcript_api import YouTubeTranscriptApi
import config
import argparse
import pandas as pd
import pdb
import time

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
    video_ids = list(video_detail_df['video'])

    for i in range(0, len(video_ids), 50):
        video_ids_sub = video_ids[i:i+1]
        transcripts = YouTubeTranscriptApi.get_transcripts(video_ids_sub, languages=['en'])
        pdb.set_trace()
        transcripts.fetch()
        if i == 0:
            break

