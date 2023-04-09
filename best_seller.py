
import os
import pandas as pd
base_src = 'Data'


# u.item 파일을 DataFrame으로 열기
u_item_src = os.path.join(base_src,'u.item')
i_cols = ['movie_id','title','release data','video release data',
          'IMDB URL','unknown','Action','Adventure','Animation',
          'Children\'s','Comedy','Crime','Documentary','Drama', 'Fantasy',
          'Film-Noir','Horror','Musical','Mystery','Romance','Sci-Fi','Thriller','War','Western']
movies = pd.read_csv(u_item_src,
                     sep='|',
                     names=i_cols,
                     encoding='latin-1'
                     
                     )
movies = movies.set_index('movie_id')
print(movies.head())


#u.data 파일을 DataFrame으로 읽기
u_data_src = os.path.join(base_src,'u.data')
r_cols = ['user_id','movie_id','rating','timestamp']
ratings = pd.read_csv(u_data_src,
                      sep = '\t',
                      names = r_cols,
                      encoding='latin-1'
                      )
ratings = ratings.set_index('user_id')
print(ratings.head())



print("인기 제품 추천")
# 인기제품 추천 방식
def recom_movie(n_items):
    movie_mean = ratings.groupby(['movie_id'])['rating'].mean()
    movie_sort = movie_mean.sort_values(ascending=False)[:n_items]
    recom_movies = movies.loc[movie_sort.index]
    recommendations = recom_movies['title']
    return recommendations
  
print(recom_movie(5))

import numpy as np

def RMSE(y_true,y_pred):
  return np.sqrt(np.mean((np.array(y_true)-np.array(y_pred))**2))


print("정확도 계산")
#정확도 계산
rmse = []
movie_mean = ratings.groupby(['movie_id'])['rating'].mean()

for user in set(ratings.index):
  y_true = ratings.loc[user]['rating']
  #best seller 방식
  y_pred = movie_mean[ratings.loc[user]["movie_id"]]
  accuracy = RMSE(y_true,y_pred)
  rmse.append(accuracy)
  
print(np.mean(rmse))