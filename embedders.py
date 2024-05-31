from data_processer import prepare_recipes_for_w2v, prepare_recipes_for_d2v
import pandas as pd
import os

from gensim.models import Word2Vec,Doc2Vec
from gensim.models import KeyedVectors

def get_model(model_type):

    if model_type == "Word2Vec":

        if os.path.exists("models/word2vec.pkl"):
            print("Saved Word2Vec model found")
            return KeyedVectors.load("models/word2vec.pkl")
        else:
            print("No Word2Vec model found, training ...")
            corpus = pd.DataFrame ( pd.read_csv("data/corpus.csv",index_col=0)["original cooking steps"] ) .dropna()
            corpus = prepare_recipes_for_w2v(corpus, "original cooking steps")
            model = Word2Vec(corpus, min_count=3, vector_size=300, workers=4, window=5)
            model.save("models/word2vec.pkl")
            return model

    elif model_type == "Doc2Vec":

        if os.path.exists("models/doc2vec.pkl"):
            print("Saved Doc2Vec model found")
            return KeyedVectors.load("models/doc2vec.pkl")
        else:
            print("No Doc2Vec model found, training ...")
            corpus = pd.DataFrame ( pd.read_csv("data/corpus.csv",index_col=0)["original cooking steps"] ) .dropna()
            corpus = prepare_recipes_for_d2v(corpus, "original cooking steps")
            model = Doc2Vec(corpus, min_count=3, vector_size=300, workers=4, window=5)
            model.save("models/doc2vec.pkl")
            return model