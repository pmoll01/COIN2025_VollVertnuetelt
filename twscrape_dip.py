import asyncio
import csv

from twscrape import API, gather
from twscrape.logger import set_log_level
from dotenv import load_dotenv
import os

##########################################################################
# damit die userdaten des verwendeten twitterkontos gezogen werden können, müssen die in einer .env im project root dir gespeichert werden



async def main():
    load_dotenv()
    api = API()  # or API("path-to.db") – default is `accounts.db`

    # ADD ACCOUNTS (for CLI usage see next readme section)

    # Option 1. Adding account with cookies (more stable)
    cookies = f"auth_token={os.environ.get('auth_token')}; ct0={os.environ.get('ct0')}"  # or '{"abc": "12", "ct0": "xyz"}'
    await api.pool.add_account(
        username=f"{os.environ.get('username')}",
        email=f"{os.environ.get('e_mail')}",
        password=f"{os.environ.get('password')}",
        email_password=f"{os.environ.get('e_mail_password')}",
        cookies=cookies)

    # # Option2. Adding account with login / password (less stable)
    # # email login / password required to receive the verification code via IMAP protocol
    # # (not all email providers are supported, e.g. ProtonMail)
    # await api.pool.add_account("user1", "pass1", "u1@example.com", "mail_pass1")
    # await api.pool.add_account("user2", "pass2", "u2@example.com", "mail_pass2")
    # await api.pool.login_all() # try to login to receive account cookies

    # API USAGE

    # # search (latest tab)
    # await gather(api.search("elon musk", limit=20))  # list[Tweet]
    # # change search tab (product), can be: Top, Latest (default), Media
    # await gather(api.search("elon musk", limit=20, kv={"product": "Top"}))
    #
    # # tweet info
    # tweet_id = 20
    # await api.tweet_details(tweet_id)  # Tweet
    # await gather(api.retweeters(tweet_id, limit=20))  # list[User]
    #
    # # Note: this method have small pagination from X side, like 5 tweets per query
    # await gather(api.tweet_replies(tweet_id, limit=20))  # list[Tweet]
    #
    # # get user by login
    # user_login = "xdevelopers"
    # await api.user_by_login(user_login)  # User
    #
    # # user info
    # user_id = 2244994945
    # await api.user_by_id(user_id)  # User
    # await gather(api.following(user_id, limit=20))  # list[User]
    # await gather(api.followers(user_id, limit=20))  # list[User]
    # await gather(api.verified_followers(user_id, limit=20))  # list[User]
    # await gather(api.subscriptions(user_id, limit=20))  # list[User]
    # await gather(api.user_tweets(user_id, limit=20))  # list[Tweet]
    # await gather(api.user_tweets_and_replies(user_id, limit=20))  # list[Tweet]
    # await gather(api.user_media(user_id, limit=20))  # list[Tweet]
    #
    # # list info
    # await gather(api.list_timeline(list_id=123456789))
    #
    # # trends
    # await gather(api.trends("news"))  # list[Trend]
    # await gather(api.trends("sport"))  # list[Trend]
    # await gather(api.trends("VGltZWxpbmU6DAC2CwABAAAACHRyZW5kaW5nAAA"))  # list[Trend]
    #
    # # NOTE 1: gather is a helper function to receive all data as list, FOR can be used as well:
    # async for tweet in api.search("elon musk"):
    #     print(tweet.id, tweet.user.username, tweet.rawContent)  # tweet is `Tweet` object
    #
    # # NOTE 2: all methods have `raw` version (returns `httpx.Response` object):
    # async for rep in api.search_raw("elon musk"):
    #     print(rep.status_code, rep.json())  # rep is `httpx.Response` object

    # change log level, default info
    set_log_level("DEBUG")

    # # Tweet & User model can be converted to regular dict or json, e.g.:
    # doc = await api.user_by_id(user_id)  # User
    # doc.dict()  # -> python dict
    # doc.json()  # -> json string

    tweets = await gather(api.user_tweets(44196397, limit=200)) #elon musk user id

    print(f"Insgesamt {len(tweets)} Tweets gefunden!")

    # Pfad zur CSV-Datei im aktuellen Ordner
    csv_path = os.path.join(os.path.dirname(__file__), "tweets_twscrape.csv")

    # Schreiben der Tweets in die CSV-Datei
    with open(csv_path, mode="w", encoding="utf-8", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["id", "created_at", "content"])  # Header

        for tweet in tweets:
            writer.writerow([tweet.id, tweet.date, tweet.rawContent])

    print(f"Tweets erfolgreich in '{csv_path}' gespeichert.")


if __name__ == "__main__":
    asyncio.run(main())
