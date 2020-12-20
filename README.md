### We can only have 10000 units per day for youtube api, which is around 100 searchs and 10000 lists etc...

### To generate more links 
```
python link_collector.py --search_word 'kids play' \
--folder_name_key_words kids_play --csv_result_dir ./video_details
```

- The search results will be saved in collected_link_json_<<kids_play>> as several json files
- The listed video details will be saved in collected_video_detail_<<kids_play>> as several json files
- The final dataframe will be saved in video details folder as csv file, 'video', 'title', 'tags','description', 'category', 'caption'

#### temporarily unused file: link_generator.py, kids_tube folder (scrapy), 