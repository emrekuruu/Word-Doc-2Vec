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
    "Chocolate Cake": process_text(["flour", "sugar", "cocoa powder", "eggs", "butter"]),
    "Apple Pie": process_text(["apples", "sugar", "flour", "butter", "cinnamon"]),
    "Caesar Salad": process_text(["lettuce", "parmesan cheese", "croutons", "lemon", "olive oil"]),
    "Greek Salad": process_text(["cucumbers", "tomatoes", "feta cheese", "olive oil", "oregano"]),
    "Caprese Salad": process_text(["tomatoes", "mozzarella", "basil leaves", "balsamic vinegar", "olive oil"]),
    "Fish Tacos": process_text(["cod", "tortillas", "cabbage", "lime", "avocado"]),
    "Shrimp Scampi": process_text(["shrimp", "linguine", "butter", "garlic", "lemon"]),
    "Vegetable Stir-Fry": process_text(["bell peppers", "broccoli", "carrots", "snap peas", "onion"]),
    "Roasted Vegetables": process_text(["potatoes", "carrots", "zucchini", "bell peppers", "onion"]),
    "Vegetable Curry": process_text(["potatoes", "cauliflower", "carrots", "bell peppers", "onion"])
}

ingredients = process_text(ingredients)

word2vec = get_model("Word2Vec")

doc2Vec = get_model("Doc2Vec")

visualize_embeddings_ingridients(model=word2vec, ingredients=ingredients, type="Word2Vec")

visualize_embeddings_ingridients(model=doc2Vec, ingredients=ingredients, type="Doc2Vec")

visualize_embeddings_recipes( recipes=recipes,model=word2vec, type="Word2Vec")

visualize_embeddings_recipes( recipes=recipes,model=doc2Vec, type="Doc2Vec")

