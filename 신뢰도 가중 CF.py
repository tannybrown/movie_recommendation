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


rating_mean = rating_matrix.mean(axis=1)

rating_bias = (rating_matrix.T - rating_mean).T


rating_binary_1 = np.array(rating_matrix >0).astype(float)
rating_binary_2 = rating_binary_1.T


# 평가한 아이템 갯수
counts = np.dot(rating_binary_1,rating_binary_2)
counts = pd.DataFrame(counts,
                      index = rating_matrix.index,
                      columns= rating_matrix.index).fillna(0)


def CF_knn_bias_sig(user_id, movie_id,neighbor_size = 0):
  if movie_id in rating_bias :
    sim_scores = user_similarity[user_id].copy()
    movie_ratings = rating_bias[movie_id].copy()
    
    no_rating = movie_ratings.isnull()
    common_counts = counts[user_id]
    low_significance = common_counts < SIG_LEVEL
    none_rating_idx = movie_ratings[no_rating | low_significance].index
    
    movie_ratings= movie_ratings.drop(none_rating_idx)
    sim_scores = sim_scores.drop(none_rating_idx)
    
    if neighbor_size == 0:
      prediction = np.dot(sim_scores,movie_ratings) / sim_scores.sum()
      prediction = prediction + rating_mean[user_id]
      
    else :
      if len(sim_scores) > MIN_RATINGS:
        neighbor_size = min(neighbor_size, len(sim_scores))
        sim_scores = np.array(sim_scores)
        movie_ratings = np.array(sim_scores)
        user_idx = np.argsort(movie_ratings)
        sim_scores = sim_scores[user_idx][-neighbor_size:]
        movie_ratings = movie_ratings[user_idx][-neighbor_size:]
        prediction = np.dot(sim_scores,movie_ratings) / sim_scores.sum()
        prediction = prediction + rating_mean[user_id]
        
      else:
        prediction = rating_mean[user_id]
        
  else: 
    prediction = rating_mean[user_id]
    
  # 혹시나 예측 값이 범위를 벗어날 경우
  if prediction < 1 :
    prediction = 1
  elif prediction > 5:
    prediction = 5
  return prediction

SIG_LEVEL = 3
MIN_RATINGS = 3
print(score(CF_knn_bias_sig,30))