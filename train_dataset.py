import pandas as pd
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import nltk

nltk.download('all')
def get_sentiment(text):
    """receives a text and produces its polarity score"""
    analyzer = SentimentIntensityAnalyzer()
    scores = analyzer.polarity_scores(text)
    return scores

def preprocess_text(text):
    """preprocesses the text before hading it to sentiment analyzer"""
    tokens = word_tokenize(text.lower())
    filtered_tokens = [token for token in tokens if token not in stopwords.words('english')]
    lemmatizer = WordNetLemmatizer()
    lemmatized_tokens = [lemmatizer.lemmatize(token) for token in filtered_tokens]
    processed_text = ' '.join(lemmatized_tokens)
    return processed_text

def pos(n):
    """used as a lambda function to get numerical values for sentiments"""
    if n >= 0: return n
    else: return 0
def neg(n):
    if n < 0: return n
    else: return 0
def net(n):
    if n == 0: return n
    else:  return 0

def query(text):
    """used as a lambda function to translate inputs"""
    translator = Translator()
    result = translator.translate(text, dest='en')
    return result.text


# read all the files
products = pd.read_parquet('products.parquet')
return_reason = pd.read_parquet('return_reasons.parquet')
returns = pd.read_parquet('returns.parquet')
reviews = pd.read_parquet('reviews.parquet')
test = pd.read_parquet('test.parquet')
reviews_return_pr = pd.read_csv('review_return_pr.csv')


#get a dataframe of returned objects
rt_kk = returns.drop_duplicates('product_id')[:200]
mask2 = returns['product_id'].isin(rt_kk['product_id'])
return_df = returns[mask2]


#apply sentiment analysis
reviews_return_pr['preprop'] = reviews_return_pr['translation'].apply(preprocess_text)
reviews_return_pr['sentiment'] = reviews_return_pr['preprop'].apply(get_sentiment)
reviews_return_pr['sentiment'] = reviews_return_pr['sentiment'].apply(lambda x: x['compound'])


#get quasi-hot encoding
reviews_return_pr['positive'] = reviews_return_pr['sentiment'].apply(pos)
reviews_return_pr['negative'] = reviews_return_pr['sentiment'].apply(neg)
reviews_return_pr['neutral'] = reviews_return_pr['sentiment'].apply(net)


gr_by_product2 = return_df.groupby(by = 'product_id').agg({'purchase_price': 'mean'})
with_mean_price = gr_by_product2.merge(return_df, on = 'product_id', how = 'right')
with_mean_price = with_mean_price.drop(['purchase_price_y', 'date_created'], axis = 1)

#savinf the results
train_df = with_mean_price.merge(gr_by_product1, on = 'product_id', how = 'left')
csv_file_path = 'train.csv'  # Desired file name
train_df.to_csv(csv_file_path, index=True)