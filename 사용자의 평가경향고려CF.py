# 사용자별로, 평가를 낮겢 주는 경향이 있을수도, 높게 주는 경향이 있을 수도 있다.
# 예를들어, 같은 3점을 줘도 각자가 느낀 바가 다를 수 있기 때문이다.
# 따라서 이러한 평가경향을 고려하는 CF를 만들어보자.

# 1. 각 사용자의 평점 평균을 계산
# 2. 평점을 각 사용자의 평균에서의 차이로 변환(평점 - 평점평균)
# 3. 평점 편차의 예측값 계산 ( 평가값 = 평점편차 * 다른 사용자 유사도)
# 4. 실제 예측값 = 평점편차 예측값 + 평점평균

import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics.pairwise import cosine_similarity


base_src = 'Data'
u_user_src = os.path.join(base_src,'u.user')
u_cols = ['user_id','age','sex','occupation','zip_code']
users = pd.read_csv(u_user_src,
                    sep = '|',
                    names = u_cols,
                    encoding = 'latin-1')
users = users.set_index('user_id')

u_item_src = os.path.join(base_src,'u.item')
i_cols = ['movie_id','title','release date','video release data',"IMDB URL",'unknown',"Action",'Adventure','Animation',
          'Children\'s','Comedy','Crime','Documentary','Drama', 'Fantasy',
          'Film-Noir','Horror','Musical','Mystery','Romance','Sci-Fi','Thriller','War','Western']
movies = pd.read_csv(u_item_src,
                     sep='|',
                     names=i_cols,
                     encoding='latin-1')
movies = movies.set_index('movie_id')

u_data_src = os.path.join(base_src, 'u.data')
r_cols = ['user_id','movie_id','rating','timestamp']
ratings = pd.read_csv(u_data_src,
                      sep='\t',
                      names = r_cols,
                      encoding = 'latin-1')

def RMSE(y_true, y_pred):
  return np.sqrt(np.mean((np.array(y_true)-np.array(y_pred))**2))


### 유사 집단의 크기를 미리 정하기 위해서 기존 score 함수에 neighbor size 인자값 추가
def score(model,neighbor_size = 0) :
  # test 데이터의 user_id와 movie_ID간의 pair를 맞춰 튜플형 원소 리스트데이터를 만듬
  id_pairs = zip(x_test['user_id'],x_test['movie_id'])
  # 모든 사용자-영화 짝에 대해서 주어진 예측모델에 의해 예측값 계산 및 리스트형 데이터 생성
  y_pred = np.array([model(user,movie,neighbor_size)for (user,movie)in id_pairs])
  # 실제 평점값
  y_true = np.array(x_test['rating'])
  return RMSE(y_true,y_pred)

x = ratings.copy()
y = ratings['user_id']

x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.25,stratify=y)

rating_matrix = x_train.pivot(index = 'user_id',columns = 'movie_id', values = "rating")


matrix_dummy = rating_matrix.copy().fillna(0)
user_similarity = cosine_similarity(matrix_dummy,matrix_dummy)
user_similarity = pd.DataFrame(user_similarity,index = rating_matrix.index, columns = rating_matrix.index)

## 사용자 평가 경향을 고려한 함수
rating_mean = rating_matrix.mean(axis=1)
# print(rating_mean)
rating_bias = (rating_matrix.T - rating_mean).T
# print(rating_bias)

def CF_knn_bias(user_id,movie_id,neighbor_size = 0):
  if movie_id in rating_bias.columns:
    sim_scores = user_similarity[user_id].copy()
    movie_ratings = rating_bias[movie_id].copy()
    none_rating_idx = movie_ratings[movie_ratings.isnull()].index
    movie_ratings = movie_ratings.drop(none_rating_idx)
    sim_scores = sim_scores.drop(none_rating_idx)
    
    if neighbor_size == 0:
      prediction = np.dot(sim_scores, movie_ratings) / sim_scores.sum()
      prediction = prediction + rating_mean[user_id]
      
    else :
      if len(sim_scores) > 1:
        neighbor_size = min(neighbor_size, len(sim_scores))
        sim_scores = np.array(sim_scores)
        movie_ratings = np.array(movie_ratings)
        user_idx = np.argsort(sim_scores)
        sim_scores = sim_scores[user_idx][-neighbor_size:]
        movie_ratings = movie_ratings[user_idx][-neighbor_size:]
        prediction = np.dot(sim_scores,movie_ratings) / sim_scores.sum()
        prediction = prediction + rating_mean[user_id]
        
      else :
        prediction = rating_mean[user_id]
  else :
    prediction = rating_mean[user_id]
  return prediction    

print(score(CF_knn_bias,30))   