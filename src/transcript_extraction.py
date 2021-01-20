from youtube_transcript_api import YouTubeTranscriptApi
import config
import argparse
import pandas as pd
import pdb
import time
import requests
import pickle
import os
import glob

CUR_FILE_DIR = os.path.dirname(os.path.realpath(__file__))
def persist_transcript(transcripts):
    '''
    Save the retrived transcripts into video caption pickle folder with key (video id) as its name
    :param transcripts: dict, key: video id, value: transcript
    :return:
    '''
    transcripts_dict, unfound = transcripts
    pd.DataFrame(unfound).to_csv(os.path.join(CUR_FILE_DIR, '../Data', 'unfound.csv'), mode='a+', index=False, header=False)
    if not os.path.exists(os.path.join(CUR_FILE_DIR, '../Data', 'video_caption_pickle')):
        os.mkdir(os.path.join(CUR_FILE_DIR, '../Data', 'video_caption_pickle'))
    for key in transcripts_dict.keys():
        pickle.dump(transcripts_dict[key], open(os.path.join(CUR_FILE_DIR, '../Data', 'video_caption_pickle', key+'.p'), 'wb+'))

def get_all_video_ids_with_transcripts(dir):

    '''
    Given a video caption pickle directory, extract the video ids appear as the file name
    :param dir:
    :return: set
    '''
    video_ids = set()
    for file_name in glob.glob(dir + '/*.p'):
        video_id = file_name.rsplit('/', 1)[1].split('.')[0]
        video_ids.add(video_id)
    return video_ids

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--video_details_file')
    parser.add_argument('--video_caption_pickle_folder', help = 'the folder name to save all retrieved transcripts')

    args = parser.parse_args()
    video_details_file = args.video_details_file
    video_caption_pickle_folder = args.video_caption_pickle_folder

    proxy_host = "proxy.crawlera.com"
    proxy_port = "8010"
    proxy_auth = config.proxy_auth
    proxies = {"https": "https://{}@{}:{}/".format(proxy_auth, proxy_host, proxy_port),
               "http": "http://{}@{}:{}/".format(proxy_auth, proxy_host, proxy_port)}

    video_detail_df = pd.read_csv(video_details_file, index_col = 0)
    video_ids = list(video_detail_df[video_detail_df['caption'] == True]['video'])
    

    ### extract the video ids appear as the file name in a folder
    video_set_with_transcripts = get_all_video_ids_with_transcripts(video_caption_pickle_folder)
    unfound_video = set(pd.read_csv(os.path.join(CUR_FILE_DIR, '../Data', 'unfound.csv'), header = None)[0])

    video_ids = list(set(video_ids) - video_set_with_transcripts - unfound_video)
    
    # video_ids = ['-TIkkGSHWeM']
    print(len(video_ids))
    for i in range(0, len(video_ids), 50):
        print('start to extract videos {0!r}'.format(str(i)))
        video_ids_sub = video_ids[i:i+50]
        print(video_ids_sub)
        transcripts = YouTubeTranscriptApi.get_transcripts(video_ids_sub, languages=['en'],
                                                           continue_after_error= True, proxies=proxies)
        persist_transcript(transcripts)
        print(str(i))
        # time.sleep(600)

