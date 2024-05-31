from gensim.models import Word2Vec
from gensim.models import KeyedVectors
import gensim
from gensim.parsing.preprocessing import preprocess_string, remove_stopwords, stem_text
import pandas as pd
from gensim.utils import simple_preprocess
from gensim.models.doc2vec import TaggedDocument

def process_text(text):
    # 3. Text preprocessing
    CUSTOM_FILTERS = [remove_stopwords, stem_text]
    tokens = [preprocess_string(doc, CUSTOM_FILTERS)[0] for doc in text]
    return tokens

def prepare_recipes_for_w2v(df, text_column):
    """
    Prepares recipe text data for Word2Vec modeling.

    Args:
    df (pd.DataFrame): DataFrame containing the recipes.
    text_column (str): Name of the column containing text.

    Returns:
    list: A list of token lists suitable for Word2Vec training.
    """
    # Tokenize text
    print('Obtain text tokens')
    tokens = [list(gensim.utils.tokenize(doc, lower=True)) for doc in df[text_column].values]
    
    #2. Bigram model training
    print('Bigram model training')
    bigram_mdl = gensim.models.phrases.Phrases(tokens, min_count=1, threshold=2)
    
    # 3. Text preprocessing
    CUSTOM_FILTERS = [remove_stopwords, stem_text]
    tokens = [preprocess_string(" ".join(doc), CUSTOM_FILTERS) for doc in tokens]
    
    print('Apply bigram model to recipe texts')
    bigrams = bigram_mdl[tokens]

    all_sentences = list(bigrams)

    return all_sentences

def prepare_recipes_for_d2v(df, text_column):
    """
    Prepares recipe text data for Doc2Vec modeling.

    Args:
    df (pd.DataFrame): DataFrame containing the recipes.
    text_column (str): Name of the column containing text.

    Returns:
    list: A list of TaggedDocument objects suitable for Doc2Vec training.
    """

    # Tokenize text
    print('Obtain text tokens')
    tokens = [list(gensim.utils.tokenize(doc, lower=True)) for doc in df[text_column].values]
    
    #2. Bigram model training
    print('Bigram model training')
    bigram_mdl = gensim.models.phrases.Phrases(tokens, min_count=1, threshold=2)
    
    # 3. Text preprocessing
    CUSTOM_FILTERS = [remove_stopwords, stem_text]
    tokens = [preprocess_string(" ".join(doc), CUSTOM_FILTERS) for doc in tokens]
    
    print('Apply bigram model to recipe texts')
    bigrams = bigram_mdl[tokens]

    all_sentences = list(bigrams)

    # Create a list of TaggedDocument, each document is tagged with its index in the dataframe
    tagged_data = [TaggedDocument(words=doc, tags=[i]) for i, doc in enumerate(all_sentences)]

    return tagged_data