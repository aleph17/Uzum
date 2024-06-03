# Uzum

ML model to predict the probability of a product being returned in the online market **link to [Uzum](https://uzum.uz/uz/)**
## Task

Product returns pose a serious problem for online marketplaces. Statistics show that approximately 15% to 40% of all online purchases are returned, which not only creates financial difficulties for businesses, but also poses serious environmental problems due to the huge amount of goods ending up in landfills. To reduce the number of returns, it is critical for sellers to be able to identify the reasons for returns as early as possible. However, this information often becomes available only after a significant number of product returns. Understanding the reasons for product returns and predicting them can go a long way in reducing this problem.

## **Target**

Develop a machine learning model that will solve a multi-class classification problem and predict the probability distribution of reasons for returning a product based on many factors, including textual customer reviews.

### **Input format**

The **link to [google drive](https://drive.google.com/drive/folders/1c9ABGWtH5xgJFIPSANEJusIxTMuwIuFD?usp=sharing)** contains 5 files:

**`return_reasons.parquet`** - dictionary file with unique reasons for returning goods. Each reason has an id and description. Total unique 5 -
`[DEFECTED, WRONG_ITEM, BAD_QUALITY, PHOTO_MISMATCH, WRONG_SIZE]`

**`reviews.parquet`** - a file with reviews of purchases on the marketplace.

Each review has:

- `order_item_id` - unique order identifier
- `product_id` - unique product identifier
- `customer_id` - unique customer identifier
- `review_text` - text customer review of the product
- `shop_id` - unique store identifier
- `rating` - rating on a scale from 1 to 5
- `date_created` - timestamp of review creation

  **`returns.parquet`** - file with returns of goods on the marketplace.

- `id` - unique identifier for returning goods
- `product_id` - unique product identifier
- `cause` - one of 5 reasons for return
- `comment` - response to return
- `date_created` - timestamp of return processing
- `order_item_id` **-** unique product order identifier
- `customer_id` - unique client identifier
- `purchase_price` - product order price

`products.parquet` - file with descriptions of products in Russian and Uzbek languages

- `product_id` **-** unique product identifier
- `category_id` ****- id of the product category
- `category_title` ****- title of the product category
- `product_description` ****- product description in Russian and Uzbek languages

  **`test.parquet`** - a file containing the same as **`returns.parquet`, except for the reason for the return. This is exactly what you have to predict.**


## **Results**
 The model uses random forest regressors to predict the probability of an object being returned for a specific reason.  Because the given reviews are in Uzbek and need to be translated into English to assess their mood (positive or negative), not all data could be translated for complete training. Thus, the performance of the model is very humble - around 60%.
