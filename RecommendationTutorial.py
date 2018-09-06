import numpy as np

from lightfm.datasets import fetch_movielens
from lightfm import LightFM


data = fetch_movielens(min_rating=4.0)

model = LightFM(loss='warp')

model.fit(data['train'], epochs=30, num_threads=2)

def sample_recommendation(model,data,user_id):
    n_users, n_items = data['train'].shape

    for id in user_id:
        known_positives = data['item_labels'][data['train'].tocsr()[id].indices]

        scores = model.predict(id,np.arange(n_items))
        top_items = data['item_labels'][np.argsort(-scores)]

        print("\n\nUser %s" % id)
        print("User Favourites:")

        for x in known_positives[:3]:
            print("%s" % x)
        print("\nRecommended:")
        for x in top_items[:3]:
            print("%s"% x)


sample_recommendation(model,data,[3,4,5])