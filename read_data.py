# u.user file dataFrame으로 읽기
import os
import pandas as pd


base_src = 'Data'
u_user_src = os.path.join(base_src,'u.user')
u_cols = ['user_id','age','sex','occupation','zip_code']
users = pd.read_csv(u_user_src,
                    sep='|',
                    names=u_cols,
                    encoding='latin-1')
users = users.set_index('user_id')
print(users.head())


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
