"""
Twitter Data Fetcher
Fetches tweets via Tweepy API v2 with fallback to synthetic demo data.
"""

import os
import time
import logging
import random
from datetime import datetime, timedelta
from typing import List, Optional

logger = logging.getLogger(__name__)

# Graceful Tweepy import
try:
    import tweepy
    _tweepy_available = True
except ImportError:
    _tweepy_available = False
    logger.warning("Tweepy not installed. Twitter API unavailable.")

# ─── Mock Tweet Templates ──────────────────────────────────────────────────────
MOCK_TWEETS = {
    "positive": [
        "Just tried {kw} for the first time and I'm absolutely blown away! 🔥 #amazing",
        "Can't stop thinking about how great {kw} is. Best decision I ever made! ✅",
        "{kw} has completely changed my workflow. So productive now! 💪 #winning",
        "Shoutout to {kw} for the incredible customer support. 5 stars! ⭐⭐⭐⭐⭐",
        "The new {kw} update is fantastic. Loving every single feature! ❤️",
        "{kw} is hands down the best in the market. Highly recommend to everyone! 👏",
        "I've been using {kw} for a year now and it just keeps getting better! 🚀",
        "Just upgraded my {kw} plan and WOW it's worth every penny! 💯",
        "{kw} customer service responded in 2 minutes. Absolutely stellar support! ✨",
        "My productivity tripled after switching to {kw}. Game changer! 🎯",
        "Loving the new {kw} interface! Clean, intuitive, and so fast! ⚡",
        "This is why I trust {kw}. Consistent quality every single time! 🙌",
        "{kw} just solved a problem I had for months. Brilliant technology! 🧠",
        "Friends kept recommending {kw} and they were 100% right. Amazing! 🎉",
        "The {kw} team really listens to users. Latest update is exactly what I needed!",
    ],
    "negative": [
        "{kw} crashed again for the 3rd time this week. So frustrated! 😤 #fail",
        "Terrible experience with {kw} support. Waited 3 days and no response. 🙄",
        "{kw} is way too expensive for what it offers. Not worth it at all! 💸",
        "Uninstalled {kw} after 2 months. Constant bugs and poor performance. 🚫",
        "Why does {kw} keep adding features nobody asked for? Just fix the bugs! 😡",
        "{kw} data breach? My account was hacked! This is unacceptable! 🔴",
        "Worst purchase of the year. {kw} simply doesn't work as advertised. 👎",
        "{kw} prices went up 40% but quality went down. Done with this service.",
        "I'm done with {kw}. Three failed attempts and they still can't get it right.",
        "The {kw} app is a disaster. Crashes every 10 minutes on my phone. 📵",
        "{kw} stole 2 hours of my life with a problem they still haven't fixed. 😤",
        "Avoid {kw} at all costs. Absolute scam company with zero accountability.",
        "Just got charged twice by {kw} and support said they'll 'look into it'. 💀",
        "{kw} is the most overrated product I've ever used. Total disappointment.",
        "Never seen such poor quality from {kw}. Complete waste of money! 🗑️",
    ],
    "neutral": [
        "Anyone have experience with {kw}? Thinking of trying it out.",
        "Just started using {kw} today. Will update after a week of testing.",
        "{kw} announced their Q3 results. Mixed performance across segments.",
        "Comparing {kw} vs competitors. Any recommendations from users?",
        "Got my {kw} order today. Haven't tried it yet.",
        "Anyone know when {kw} releases their next update?",
        "Using {kw} for a project. It's doing what it's supposed to do.",
        "Read an interesting article about {kw}'s expansion plans.",
        "{kw} is having a sale this weekend. Might check it out.",
        "My colleague recommended {kw}. Downloading it to try.",
        "The {kw} conference was informative. Lots of new announcements.",
        "Switched to {kw} last month. Still getting used to the interface.",
        "{kw} has some interesting features but I'm still exploring.",
        "Curious about {kw} API documentation. Need to integrate it.",
        "Just renewed my {kw} subscription for another year.",
    ],
}


class TwitterFetcher:
    """
    Fetches tweets from Twitter API v2 (Tweepy) or generates synthetic data.
    """

    def __init__(self, use_mock: bool = False):
        self.use_mock = use_mock
        self.client = None

        if not use_mock and _tweepy_available:
            self._init_client()

    def _init_client(self):
        """Initialize Tweepy v2 client with Bearer Token."""
        bearer_token = os.getenv("TWITTER_BEARER_TOKEN")
        if not bearer_token:
            logger.warning("TWITTER_BEARER_TOKEN not set. Falling back to mock data.")
            self.use_mock = True
            return
        try:
            self.client = tweepy.Client(
                bearer_token=bearer_token,
                wait_on_rate_limit=True,
            )
            logger.info("Twitter API client initialized successfully.")
        except Exception as e:
            logger.error(f"Failed to initialize Twitter client: {e}")
            self.use_mock = True

    def fetch(self, keyword: str, count: int = 100) -> List[str]:
        """
        Fetch tweets for a keyword.
        Returns list of tweet text strings.
        """
        if self.use_mock or self.client is None:
            return self._generate_mock_tweets(keyword, count)
        return self._fetch_real_tweets(keyword, count)

    def _fetch_real_tweets(self, keyword: str, count: int) -> List[str]:
        """Fetch real tweets using Twitter API v2."""
        tweets = []
        try:
            query = f"{keyword} -is:retweet lang:en"
            # API v2 allows max 100 per request
            per_request = min(count, 100)

            response = self.client.search_recent_tweets(
                query=query,
                max_results=per_request,
                tweet_fields=["text", "created_at", "lang"],
            )

            if response.data:
                tweets = [tweet.text for tweet in response.data]
                logger.info(f"Fetched {len(tweets)} real tweets for '{keyword}'")
            else:
                logger.warning(f"No tweets found for '{keyword}'. Using mock data.")
                tweets = self._generate_mock_tweets(keyword, count)

        except tweepy.TooManyRequests:
            logger.warning("Twitter rate limit hit. Using mock data.")
            tweets = self._generate_mock_tweets(keyword, count)
        except tweepy.Unauthorized:
            logger.error("Twitter API unauthorized. Check Bearer Token.")
            tweets = self._generate_mock_tweets(keyword, count)
        except Exception as e:
            logger.error(f"Twitter fetch error: {e}")
            tweets = self._generate_mock_tweets(keyword, count)

        return tweets[:count]

    def _generate_mock_tweets(self, keyword: str, count: int) -> List[str]:
        """Generate realistic synthetic tweets for demo/testing."""
        random.seed(42)

        # Distribute: 45% positive, 30% negative, 25% neutral
        n_pos = int(count * 0.45)
        n_neg = int(count * 0.30)
        n_neu = count - n_pos - n_neg

        tweets = []

        pos_templates = MOCK_TWEETS["positive"] * (n_pos // len(MOCK_TWEETS["positive"]) + 1)
        neg_templates = MOCK_TWEETS["negative"] * (n_neg // len(MOCK_TWEETS["negative"]) + 1)
        neu_templates = MOCK_TWEETS["neutral"] * (n_neu // len(MOCK_TWEETS["neutral"]) + 1)

        tweets += [t.replace("{kw}", keyword) for t in random.sample(pos_templates, n_pos)]
        tweets += [t.replace("{kw}", keyword) for t in random.sample(neg_templates, n_neg)]
        tweets += [t.replace("{kw}", keyword) for t in random.sample(neu_templates, n_neu)]

        # Add some noise/variation
        noise_tweets = [
            f"Reading about {keyword} right now. Interesting stuff.",
            f"My friend loves {keyword}. I'm on the fence about it.",
            f"{keyword} just launched something new. Will check it out later.",
            f"People are really talking about {keyword} today.",
            f"Saw a review of {keyword}. Deciding whether to try it.",
        ]
        tweets += noise_tweets[:max(0, count - len(tweets))]

        random.shuffle(tweets)
        logger.info(f"Generated {len(tweets)} mock tweets for '{keyword}'")
        return tweets[:count]