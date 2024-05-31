from embedders import get_model
from visualizer import visualize_embeddings
from data_processer import process_text
ingredients = [
    # Meat
    "brisket", "lamb", "turkey", "pork", "beef", "salmon", "fillet", 
    "salmon_skin", "fish", "trout",

    # Vegetables
    "lettuce", "salad", "cucumber", "avocado", "onion", 
    "carrot", "leek", "mushroom", "shallot", "garlic",

    # Grains
    "noodles", "rice", "couscous", "quinoa", "basmati_rice", "wild_rice", 
    "cereal", "oat",

    # Desserts
    "cake", "brownie", "cheesecake", "popcorn", "pretzel"
]

ingredients = process_text(ingredients)

print(ingredients)

word2vec = get_model("Word2Vec")

doc2Vec = get_model("Doc2Vec")

visualize_embeddings(model=word2vec, ingredients=ingredients, type="Word2Vec")

visualize_embeddings(model=doc2Vec, ingredients=ingredients, type="Doc2Vec")

