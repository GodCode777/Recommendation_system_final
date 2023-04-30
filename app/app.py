import pickle
import pandas as pd
import numpy as np
from xgboost import XGBRanker
from operator import itemgetter 
from recommend import get_content_based_recommendations, get_matrix_factorized_recommendations
from flask import Flask, render_template, request, redirect, url_for
from werkzeug.datastructures import ImmutableMultiDict


app = Flask(__name__)

# Load the user data from the CSV file
print('Loading Data...')
users_df = pd.read_csv('users.csv')
ratings_file = pd.read_csv('/Users/meetkantesaria/Documents/Meet/DA-IICT/RS/final_assignment/app/source/Digital_Music_small.csv', header=None)

with open('/Users/meetkantesaria/Documents/Meet/DA-IICT/RS/final_assignment/app/source/item_vectors_v2.pickle', 'rb') as handle:
    item_vec_pickle = pickle.load(handle)

item_ids = np.array(list(item_vec_pickle.keys()))
item_vectors = np.array(list(item_vec_pickle.values()))

with open('/Users/meetkantesaria/Documents/Meet/DA-IICT/RS/final_assignment/app/source/matrix_factorization_model_v1.pickle', 'rb') as handle:
    matrix_pickle = pickle.load(handle)

xgb_ranker = XGBRanker()
xgb_ranker.load_model("/Users/meetkantesaria/Documents/Meet/DA-IICT/RS/final_assignment/app/source/l2r_xgb_v1.json")

title_df = pd.read_csv('/Users/meetkantesaria/Documents/Meet/DA-IICT/RS/final_assignment/app/source/title.csv')
title_df = title_df.drop('Unnamed: 0', axis=1)

user_keys = list(matrix_pickle['user_dict'].keys())

persistant_profile = []
ephemeral_pos_profile = []
ephemeral_neg_profile = []
reviewer_id = None

# Define the route for the login page
@app.route('/', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        global reviewer_id 
        reviewer_id = request.form['reviewer_id']
        # Check if the reviewer_id exists in the user data
        if reviewer_id not in user_keys:
            df = pd.read_csv('/Users/meetkantesaria/Documents/Meet/DA-IICT/RS/final_assignment/app/source/popilarity_based.csv')[['asin', 'title']]
            render_template('reviews.html', reviewer_id=reviewer_id, user_reviews=df)
        try:
            items_interacted = list(ratings_file[ratings_file[1] == reviewer_id][3])
            persistant_profile.extend(items_interacted)
            
            temp = persistant_profile + ephemeral_pos_profile
            items = {i:'pos' for i in temp}

            for k in ephemeral_neg_profile:
                items[k] = 'neg'
            
            rec_items_content = get_content_based_recommendations(items,item_vectors, item_ids, 10)
            rec_items_matrix = get_matrix_factorized_recommendations(matrix_pickle, user_keys, reviewer_id, 10)
            
            rank_vectors = []
            for item in list(rec_items_matrix) + list(rec_items_content):
                try:
                    rank_vectors.append(item_vec_pickle[item])
                except:
                    pass
            
            rank_df = pd.DataFrame(rank_vectors)
            rank = np.argsort(xgb_ranker.predict(rank_df))
            ranked_items = itemgetter(*list(rank))(list(rec_items_matrix) + list(rec_items_content))

            ranked_items = title_df[title_df['asin'].isin(ranked_items)]
            return render_template('reviews.html', reviewer_id=reviewer_id, user_reviews=ranked_items)
        except:
            return "ReviewerID not found"
    else:
        return render_template('login.html')
    
@app.route('/rec', methods=['GET', 'POST'])
def rec():
    global reviewer_id
    if request.method == 'POST':
        print(request.form['action'])
        if request.form['action'] == 'next':
            d = ImmutableMultiDict(request.form)
            for i in list(d.lists()):
                if 'pos' in i[0]:
                    ephemeral_pos_profile.extend(i[1])
                elif 'neg' in i[0]:
                    ephemeral_neg_profile.extend(i[1])
            
            temp = persistant_profile + ephemeral_pos_profile
            items = {i:'pos' for i in temp}

            for k in ephemeral_neg_profile:
                items[k] = 'neg'
            
            rec_items_content = get_content_based_recommendations(items,item_vectors, item_ids, 10)
            rec_items_matrix = get_matrix_factorized_recommendations(matrix_pickle, user_keys, reviewer_id, 10)
            rank_vectors = []
            for item in list(rec_items_matrix) + list(rec_items_content):
                try:
                    rank_vectors.append(item_vec_pickle[item])
                except:
                    pass
            
            rank_df = pd.DataFrame(rank_vectors)
            rank = np.argsort(xgb_ranker.predict(rank_df))
            ranked_items = itemgetter(*list(rank))(list(rec_items_matrix) + list(rec_items_content))

            ranked_items = title_df[title_df['asin'].isin(ranked_items)]
            return render_template('reviews.html', reviewer_id=reviewer_id, user_reviews=ranked_items)
        else:
            return render_template('login.html')

        return render_template('reviews.html')

# Define a function to get the previously reviewed items by the user
# def get_user_reviews(reviewer_id):
#     reviews_df = pd.read_csv('reviews.csv')
#     user_reviews = reviews_df[reviews_df['reviewerID'] == reviewer_id]
#     return user_reviews

# Define the route for the user page
# @app.route('/user/<reviewer_id>', methods=['GET'])
# def user(reviewer_id):
#     user_reviews = get_user_reviews(reviewer_id)
#     return render_template('reviews.html', reviewer_id=reviewer_id, user_reviews=user_reviews)

if __name__ == '__main__':
    app.run(debug=True)
