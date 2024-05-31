from embedders import get_model
from visualizer import visualize_embeddings_recipes, visualize_embeddings_ingridients
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

recipes = {
    "Chocolate Cake": ["flour", "sugar", "cocoa powder", "eggs", "butter"],
    "Apple Pie": ["apples", "sugar", "flour", "butter", "cinnamon"],
    "Grilled Salmon": ["salmon", "olive oil", "lemon", "garlic", "dill"],
    "Fish Tacos": ["cod", "tortillas", "cabbage", "lime", "avocado"],
    "Shrimp Scampi": ["shrimp", "linguine", "butter", "garlic", "lemon"],
    "Vegetable Stir-Fry": ["bell peppers", "broccoli", "carrots", "snap peas", "onion"],
    "Roasted Vegetables": ["potatoes", "carrots", "zucchini", "bell peppers", "onion"],
    "Vegetable Curry": ["potatoes", "cauliflower", "carrots", "bell peppers", "onion"]
}

ingredients = process_text(ingredients)

print(ingredients)

word2vec = get_model("Word2Vec")

doc2Vec = get_model("Doc2Vec")

visualize_embeddings_ingridients(model=word2vec, ingredients=ingredients, type="Word2Vec")

visualize_embeddings_ingridients(model=doc2Vec, ingredients=ingredients, type="Doc2Vec")

visualize_embeddings_recipes( recipes=recipes,model=word2vec, type="Word2Vec")

visualize_embeddings_recipes( recipes=recipes,model=doc2Vec, type="Doc2Vec")

