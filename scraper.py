import snscrape.modules.twitter as sntwitter

class Scraper:
    def __init__(self):
        self.tweets = []
        
    def scraping(self, keyword, tweets):
    # Setting variables to be used below
        tweets = int(tweets)

        # Creating list to append tweet data to
        tweets_list2 = []
        
        self.tweets = sntwitter.TwitterSearchScraper(keyword, 'lang:"id"').get_items()
        
        for i,tweet in enumerate(self.tweets):
            if i>tweets:
                break
            tweets_list2.append([tweet.date, tweet.id, tweet.content, tweet.lang, tweet.user.username])
        
        return tweets_list2