import praw
import pandas as pd
from config import settings

def fetch_reddit_data(subreddit_name, limit=1000):
    reddit = praw.Reddit(
        client_id = settings.REDDIT_CLIENT_ID,
        client_secret = settings.REDDIT_CLIENT_SECRET,
        user_agent= settings.USER_AGENT
    )

    posts = []
    subreddit = reddit.subreddit(subreddit_name)

    for submission in subreddit.hot(limit=limit):
        posts.append({
            "id":submission.id,
            "text":submission.selftext,
            "title":submission.title,
            "score":submission.score,
            "created_utc":submission.created_utc
        })
    
    df = pd.DataFrame(posts)
    df.to_csv("data/raw/reddit_data.csv", index=False)
    return df