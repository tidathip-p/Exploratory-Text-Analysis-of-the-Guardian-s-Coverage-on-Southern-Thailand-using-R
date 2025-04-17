### Step 1: Load Required Libraries ###

# Install the packages 
install.packages(c("rvest", "dplyr", "stringr", "tm", "tidytext", "ggplot2", "tidyr", "topicmodels
", "SnowballC"))

# Load the libraries
library(rvest)
library(dplyr)
library(stringr)
library(tm)
library(tidytext)
library(ggplot2)
library(tidyr)
library(topicmodels)
library(SnowballC)

### Step 2: Scrape the Web Pages ###

# Function to scrape an article
scrape_article <- function(url) {
  webpage <- read_html(url)
  
  # Extract title
  title <- webpage %>%
    html_node("title") %>%
    html_text(trim = TRUE)
  
  # Extract date 
  date <- webpage %>%
    html_node("meta[property='article:published_time']") %>%
    html_attr("content") %>%
    str_extract("\\d{4}-\\d{2}-\\d{2}")  # Extract date format
  
  # Extract article content 
  content <- webpage %>%
    html_nodes("p") %>%
    html_text(trim = TRUE) %>%
    paste(collapse = " ")
  
  # Return as a data frame
  return(data.frame(title = title, date = date, content = content, url = url, stringsAsFactors = FALSE))
}

# Apply the function to all URLs
urls <- c(
  "https://www.theguardian.com/world/2004/jul/25/jasonburke.theobserver",
  "https://www.theguardian.com/world/2004/oct/27/thailand",
  "https://www.theguardian.com/world/2004/oct/26/thailand?CMP=gu_com",
  "https://www.theguardian.com/world/2004/may/06/worlddispatch.johnaglionby",
  "https://www.theguardian.com/theguardian/2005/feb/28/theeditorpressreview.thailand",
  "https://www.theguardian.com/world/2005/mar/28/thailand",
  "https://www.theguardian.com/world/2006/sep/04/thailand",
  "https://www.theguardian.com/world/2006/jan/28/comment.simontisdall",
  "https://www.theguardian.com/world/2006/sep/21/thailand2",
  "https://www.theguardian.com/world/2008/jan/14/thailand.ianmackinnon",
  "https://www.theguardian.com/commentisfree/2009/dec/01/thailand-malay-muslims",
  "https://www.theguardian.com/world/2013/feb/28/thailand-muslim-separatists-peace-talks",
  "https://www.theguardian.com/world/2015/may/15/string-of-bomb-attacks-in-thailands-restive-south-injure-18-people",
  "https://www.theguardian.com/world/2015/jul/11/seven-dead-and-dozens-injured-in-attacks-in-thailands-south",
  "https://www.theguardian.com/world/2015/apr/11/koh-samui-car-bomb-seven-people-hurt-on-thai-tourist-island",
  "https://www.theguardian.com/world/2015/nov/13/four-die-in-southern-thailand-bombing",
  "https://www.theguardian.com/world/2015/aug/18/bangkok-bomb-not-in-keeping-with-insurgent-attacks-says-army-chief",
  "https://www.theguardian.com/global-development/2016/aug/24/bridging-language-divide-thailand-muslim-patani-malay-schools-ethnic-tension",
  "https://www.theguardian.com/world/2016/aug/24/two-bombs-explode-in-thai-beach-resort-of-pattani",
  "https://www.theguardian.com/world/2016/aug/19/thai-police-hunt-one-man-confusing-inquiry-into-bombings",
  "https://www.theguardian.com/world/2016/aug/11/one-dead-and-20-injured-in-two-bomb-blasts-in-hua-hin-thailand",
  "https://www.theguardian.com/world/live/2016/aug/12/thailand-explosions-resort-towns-phuket-hua-hin?filterKeyEvents=false&page=with:block-57ad4e6fe4b024cdec3d6c41",
  "https://www.theguardian.com/world/2016/aug/12/thailand-attacks-who-are-likely-perpetrators",
  "https://www.theguardian.com/world/2019/nov/06/dead-killed-in-biggest-attack-in-thailands-restive-south-in-years",
  "https://www.theguardian.com/global-development/2021/aug/17/the-woman-on-a-mission-to-expose-torture-in-thailands-troubled-south"
  )

# Scrape all articles
articles <- do.call(rbind, lapply(urls, scrape_article))

# View the scraped data
View(articles)

### Step 3: Data Preprocessing ###

# Function to clean and preprocess text
clean_text <- function(text) {
  text <- tolower(text)  # Convert to lowercase
  text <- removePunctuation(text)  # Remove punctuation
  text <- removeNumbers(text)  # Remove numbers
  text <- removeWords(text, stopwords("en"))  # Remove stopwords
  text <- wordStem(text, language = "en")  # Perform stemming
  text <- stripWhitespace(text)  # Remove extra whitespace
  return(text)
}

# Apply the cleaning function to the article content
articles$content_clean <- sapply(articles$content, clean_text)

# View the cleaned content
head(articles$content_clean)

### Step 4: Word Frequency ###

# Tokenize the cleaned content
articles_tidy <- articles %>%
  unnest_tokens(word, content_clean)

# Count word frequencies
word_freq <- articles_tidy %>%
  count(word, sort = TRUE)

# View the most frequent words
head(word_freq, 22)

# Visualize the most common words 
word_freq %>%
  filter(n > 30) %>%
  ggplot(aes(x = reorder(word, n), y = n)) +
  geom_col() +
  coord_flip() +
  labs(title = "Most Common Words in Articles", x = "Word", y = "Frequency")

### Step 5: Sentiment Analysis ###

# Load sentiment lexicon
sentiment_lexicon <- get_sentiments("bing")

# Perform sentiment analysis
article_sentiments <- articles_tidy %>%
  inner_join(sentiment_lexicon, by = "word") %>%
  count(url, sentiment) %>%
  pivot_wider(names_from = sentiment, values_from = n, values_fill = list(n = 0)) %>%
  mutate(sentiment_score = positive - negative)

# View sentiment scores by article
article_sentiments

# Visualize sentiment scores 
article_sentiments %>%
  ggplot(aes(x = url, y = sentiment_score, fill = factor(sentiment_score > 0))) +
  geom_col() +
  coord_flip() +
  labs(title = "Sentiment Scores by Article", x = "Article", y = "Sentiment Score")

### Step 6: Topic Modeling ###

# Create a document-term matrix
dtm <- articles_tidy %>%
  count(url, word) %>%
  cast_dtm(url, word, n)

# Fit the LDA model with a chosen number of topics (k = 3)
lda_model <- LDA(dtm, k = 3, control = list(seed = 1234))

# Extract topics
topics <- tidy(lda_model, matrix = "beta")

# View top terms in each topic
top_terms <- topics %>%
  group_by(topic) %>%
  top_n(10, beta) %>%
  ungroup() %>%
  arrange(topic, -beta)

View(top_terms)

# Visualize top terms in each topic 
top_terms %>%
  ggplot(aes(x = reorder_within(term, beta, topic), y = beta, fill = factor(topic))) +
  geom_col(show.legend = FALSE) +
  facet_wrap(~ topic, scales = "free") +
  coord_flip() +
  scale_x_reordered() +
  labs(title = "Top Terms in Each Topic", x = "Term", y = "Beta")
