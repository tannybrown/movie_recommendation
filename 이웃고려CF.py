# 이웃 만드는 방법
# KNN 방법 or THresholding 방법을 이용함.

import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

#### 데이터 불러오기 및 데이터 셋 만들기 ####

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

#### train set 의 모든 가능한 사용자 pair의 cosine similarities 계산
# 코사인 유사도를 계산하는 사이킷런 라이브러리
from sklearn.metrics.pairwise import cosine_similarity
# 코사인 유사도 구하기 위해서 rating 복사하고, 게산 시 NaN값 에러를 대비하기 위해서 결측치를 0으로 대체
matrix_dummy = rating_matrix.copy().fillna(0)
# 모든 사용자간 코사인 유사도 구함
user_similarity = cosine_similarity(matrix_dummy,matrix_dummy)

# 필요한 값 조회를 위해 인덱스 및 컬럼명 지정
user_similarity = pd.DataFrame(user_similarity,
                               index=rating_matrix.index,
                               columns= rating_matrix.index)



def CF_knn(user_id, movie_id, neighbor_size = 0) :
  if movie_id in rating_matrix.columns :
    sim_scores = user_similarity[user_id].copy()
    movie_ratings = rating_matrix[movie_id].copy()
    none_rating_idx = movie_ratings[movie_ratings.isnull()].index
    movie_ratings = movie_ratings.dropna()
    sim_scores = sim_scores.drop(none_rating_idx)
    
    if neighbor_size == 0 :
      mean_rating = np.dot(sim_scores,movie_ratings / sim_scores.sum())
      
    else :
      if (len(sim_scores) >1) :
        neighbor_size = min(neighbor_size,len(sim_scores))
        sim_scores = np.array(movie_ratings)
        movie_ratings = np.array(movie_ratings)
        user_idx = np.argsort(sim_scores)
        sim_scores = sim_scores[user_idx][-neighbor_size:]
        movie_ratings = movie_ratings[user_idx][-neighbor_size:]
        mean_rating = np.dot(sim_scores,movie_ratings)/sim_scores.sum()
      else :
        mean_rating = 3.0
  else :
    mean_rating = 3.0
    
  return mean_rating

# # 정확도 계산
# print(score(CF_knn,neighbor_size=30))



## 실제 주어진 사용자에 대해 추천을 받는 기능 구현 ###
rating_matrix = ratings.pivot_table(values = 'rating',
                                    index = 'user_id',
                                    columns='movie_id')
matrix_dummy = rating_matrix.copy().fillna(0)
user_similarity = cosine_similarity(matrix_dummy,matrix_dummy)
user_similarity = pd.DataFrame(user_similarity,
                               index=rating_matrix.index,
                               columns=rating_matrix.index)

def recom_movie(user_id,n_items,neighbor_size =30) :
  user_movie = rating_matrix.loc[user_id].copy()
  
  for movie in rating_matrix.columns :
    if pd.notnull(user_movie.loc[movie]) :
      user_movie.loc[movie] = 0
      
    else :
      user_movie.loc[movie] = CF_knn(user_id,movie,neighbor_size)
      
  movie_sort = user_movie.sort_values(ascending = False)[:n_items]
  recom_movies = movies.loc[movie_sort.index]
  recommendations = recom_movies['title']
  return recommendations



print(recom_movie(user_id = 729,n_items = 5,neighbor_size=30))