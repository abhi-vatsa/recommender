import numpy as np
import pandas as pd
import pickle

with open("nmf_model.pkl", "rb") as file:
        nmf_model = pickle.load(file)
        
def recommend_nmf(user_query,model = nmf_model, k=10):
    """
    Filters and recommends the top k movies for any given input query based
    on a trained NMF model.
    Returns a list of k movie ids.
    """
    

    movies_title_df = pd.read_csv("./data/movie_title.csv", index_col=0)

    # 1. construct new_user-item dataframe given the query
    new_user_df = pd.DataFrame(
        user_query, columns=movies_title_df["title"], index=["new_user"]
    ).fillna(0)

    # 2. scoring: calculate the score with the NMF model
    Q_matrix = nmf_model.components_
    Q = pd.DataFrame(Q_matrix)
    P_new_user_matrix = nmf_model.transform(new_user_df)
    R_hat_new_user_matrix = np.dot(P_new_user_matrix, Q)
    R_hat_new_user = pd.DataFrame(
        data=R_hat_new_user_matrix, columns=movies_title_df["title"], index=["new_user"]
    )

    # 3. ranking: filter out movies already seen by the user
    R_hat_new_user_filtered = R_hat_new_user.drop(user_query.keys(), axis=1)
    ranked = R_hat_new_user_filtered.T.sort_values(
        by=["new_user"], ascending=False
    ).index.to_list()

    # 4. return the top-k highest rated movie ids or titles
    recommendation = ranked[:3]
    return recommendation


def recommend_col(user_query, k=10):
    """
    Filters and recommends the top k movies for any given input query
    based on a trained neigbourhood based collaborative filtering model.
    Returns a list of k movie ids.
    """
    with open("knn.pkl", "rb") as file:
        neighbors_model = pickle.load(file)

    R_df = pd.read_csv("./data/movie_matrix.csv", index_col=0)
    movies_title_df = pd.read_csv("./data/movie_title.csv", index_col=0)

    R_df.columns = movies_title_df["title"]

    # 1. construct new_user-item dataframe given the query
    new_user_df = pd.DataFrame(
        user_query, columns=movies_title_df["title"], index=["new_user"]
    ).fillna(0)

    # 2. scoring
    _, neighbor_ids = neighbors_model.kneighbors(
        new_user_df, n_neighbors=15, return_distance=True
    )

    # 3. ranking
    neighborhood = R_df.iloc[neighbor_ids[0]]
    neighborhood_filtered = neighborhood.drop(user_query.keys(), axis=1)
    df_score = neighborhood_filtered.sum()
    df_score_ranked = df_score.sort_values(ascending=False).index.tolist()
    recommendation = df_score_ranked[:3]

    return recommendation  # , df_score.sort_values(ascending=False)
