### We can only have 10000 units per day for youtube api, which is around 100 searchs and 10000 lists etc...

### To generate more links 
```
python link_collector.py --search_word 'kids play' \
--folder_name_key_words kids_play --csv_result_dir ./video_details
```

- The search results will be saved in collected_link_json_<<kids_play>> as several json files
- The listed video details will be saved in collected_video_detail_<<kids_play>> as several json files
- The final dataframe will be saved in video details folder as csv file, 'video', 'title', 'tags','description', 'category', 'caption'

### Dataset 
#### Video details

| video | title | tags | description | category | caption |
| :_____: | :_____: | :_____: | :_____: | :_____: | :_____: |
| id | str | list | str | str | boolean |

#### video_caption_pickle

Note: the transcript with () in the text means a notation but a true transcript, for example: cow mooing..
file name => video_id.p
[{
    'text': "...",
    'start': float (time),
    'duration': float(end - start)
},
{
    'text': "...",
    'start': float (time),
    'duration': float(end - start)
}, 
....]

### temporarily unused file: link_generator.py, kids_tube folder (scrapy), 

### Notes about the transformers
- BERT is too large, DistilBERT is probably a better choice, a “distilled” version of BERT that is smaller and faster while retaining most of BERT’s accuracy
- For BERT-based models, learning rates between 2e-5 and 5e-5 generally work well across a wide range of datasets. 
- ktrain: a wraper package for huggingface transformer