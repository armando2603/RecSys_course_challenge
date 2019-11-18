"""
Created on 01/10/19

@author: Giuseppe Serna
"""
from tqdm import tqdm
from DataManager.DataManager import DataManager
import RecommenderAlgorithms as Recommender
import pandas as pd

Data = DataManager()
MyRecommender = Recommender.RandomRecommender()
MyRecommender.fit(Data.get_urm())
target_playlist = pd.read_csv('Data/target_playlists.csv')
target_playlist = list(target_playlist['playlist_id'])

recommended_list = []
for playlist in tqdm(target_playlist):
    recommended_items = MyRecommender.recommend(playlist, 10)
    items_strings = ' '.join([str(i) for i in recommended_items])
    recommended_list.append(items_strings)

submission = pd.DataFrame(list(zip(target_playlist, recommended_list)), columns=['playlist_id', 'track_ids'])
submission.to_csv('Data/my_first_submission.csv', index=False)




