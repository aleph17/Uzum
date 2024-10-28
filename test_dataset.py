import pandas as pd
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from googletrans import Translator


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

# reading all files
products = pd.read_parquet('products.parquet')
return_reason = pd.read_parquet('return_reasons.parquet')
returns = pd.read_parquet('returns.parquet')
reviews = pd.read_parquet('reviews.parquet')
test = pd.read_parquet('test.parquet')


# take 30 product_id from test and its reviews from reviews dataframe
unique_pds =returns.drop_duplicates('product_id')
mask = ~(test['product_id'].isin(unique_pds['product_id']))
test_kk = test[mask].drop_duplicates('product_id')[:30]
mask2 = reviews['product_id'].isin(test_kk['product_id'])
review_test = reviews[mask2]
test_df = test[test['product_id'].isin(test_kk['product_id'])]


#getting rid of N/A values
review_test = review_test.dropna(subset = 'review_text')
review_test = review_test[review_test['review_text'] != '']

#translate the inputs into english
review_test['translation'] = review_test['review_text'].apply(query)

#apply sentiment analysis
review_test['preprop'] = review_test['translation'].apply(preprocess_text)
review_test['sentiment'] =review_test['preprop'].apply(get_sentiment)
review_test['sentiment'] = review_test['sentiment'].apply(lambda x: x['compound'])


#get quasi-hot encoding
review_test['neutral'] = review_test['sentiment'].apply(net)
review_test['negative'] = review_test['sentiment'].apply(neg)
review_test['positive'] = review_test['sentiment'].apply(pos)


dic ={'rating': 'mean', 'sentiment':"mean", 'positive':'mean', 'negative': 'mean', 'neutral': 'mean'}
gr = review_test.groupby(by = 'product_id').agg(dic).reset_index(drop=False)

gr_2 = test_df.groupby(by = 'product_id').agg({'purchase_price': 'mean'})
with_mean_price = gr_2.merge(test_df, on = 'product_id', how = 'right')
with_mean_price = with_mean_price.drop('purchase_price_y', axis = 1)

#savinf the results
test_ = with_mean_price.merge(gr, on = 'product_id', how = 'left')
csv_file_path = 'test_.csv'  # Desired file name
test_.to_csv(csv_file_path, index=True)