# u.user file dataFrame으로 읽기
import os
import pandas as pd
import numpy as np

base_src = 'Data'
u_user_src = os.path.join(base_src,'u.user')
u_cols = ['user_id','age','sex','occupation','zip_code']
users = pd.read_csv(u_user_src,
                    sep='|',
                    names=u_cols,
                    encoding='latin-1')
# users = users.set_index('user_id')

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
# movies = movies.set_index('movie_id')

#u.data 파일을 DataFrame으로 읽기
u_data_src = os.path.join(base_src,'u.data')
r_cols = ['user_id','movie_id','rating','timestamp']
ratings = pd.read_csv(u_data_src,
                      sep = '\t',
                      names = r_cols,
                      encoding='latin-1'
                      )
# ratings = ratings.set_index('user_id')





#ratings DataFrame에서 timestamp 제거
ratings = ratings.drop('timestamp',axis = 1)
movies = movies[['movie_id','title']]


# 데이터 train, test set 분리
from sklearn.model_selection import train_test_split
x = ratings.copy()
y = ratings['user_id']

x_train,x_test,y_train,y_test = train_test_split(x,y,
                                                 test_size = 0.25,
                                                 stratify=y)

# 정확도(RMSE)를 계산하는 함수
def RMSE(y_true,y_pred):
  return np.sqrt(np.mean((np.array(y_true)-np.array(y_pred))**2))


#모덷별 RMSE 계산하는 함수
def score(model):
  id_pairs = zip(x_test['user_id'],x_test['movie_id'])
  y_pred = np.array([model(user,movie)for (user,movie) in id_pairs])
  y_true = np.array(x_test['rating'])
  return RMSE(y_true,y_pred)

#best_seller 함수를 이용한 정확도 계산
train_mean = x_train.groupby(['movie_id'])['rating'].mean()
def best_seller(user_id,movie_id):
  try :
    rating = train_mean[movie_id]
    
  except:
    rating = 3.0
    
  return rating
print("모델별 RMSE")
print(score(best_seller))


# 성별에 따른 예측값 계산
merged_ratings = pd.merge(x_train,users)

users = users.set_index('user_id')

g_mean = merged_ratings[['movie_id','sex','rating']].groupby(['movie_id',
                                                              'sex'])['rating'].mean()

rating_matrix = x_train.pivot(index = 'user_id',
                              columns = 'movie_id',
                              values = 'rating')

print(rating_matrix)


#Gender 기준 추천
def cf_gender(user_id,movie_id):
  if movie_id in rating_matrix.columns:
    gender = users.loc[user_id]['sex']
    if gender in g_mean[movie_id].index:
      gender_rating = g_mean[movie_id][gender]
    else :
      gender_rating = 3.0
  else :
    gender_rating = 3.0
  return gender_rating

print("gender 기준 평가")
print(score(cf_gender))