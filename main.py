import os
import re
import json
import time
import csv
from collections import defaultdict

import cloudscraper
from bs4 import BeautifulSoup
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import pickle
from datetime import date

# ----------------------------
# Config
# ----------------------------
USER_ID = "107780257626128497"  # Truth Social numeric user id
LIMIT = 200
TARGET_USABLE_POSTS = 50
EXCLUDE_REPLIES = "false"

# Pacing to reduce Truth Social rate limiting (HTTP 429)
PAGE_DELAY_SECONDS = float(os.getenv("TS_PAGE_DELAY_SECONDS", "1.0"))
RATE_LIMIT_FLOOR_SECONDS = float(os.getenv("TS_RATE_LIMIT_FLOOR_SECONDS", "15"))


URL_RE = re.compile(r"https?://\S+")

def normalize_post(text: str) -> str:
    """Remove URLs + normalize whitespace. Keeps the poster's words."""
    text = URL_RE.sub("", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


# ----------------------------
# VADER sentiment + topic→company mapping
# ----------------------------
analyzer = SentimentIntensityAnalyzer()

TOPIC_COMPANY = {
    "oil": "Chevron",
    "energy": "Exxon Mobil",
    "gas": "Exxon Mobil",
    "military": "Lockheed Martin",
    "defense": "Lockheed Martin",
    "aircraft": "Boeing",
    "jet": "Lockheed Martin",
    "media": "Fox Corporation",
    "news": "Fox Corporation",
    "youtube": "Alphabet",
    "google": "Alphabet",
    "water": "Xylem",
    "sewage": "Xylem",
    "infrastructure": "AECOM",
    "construction": "Caterpillar",
    "border": "Palantir",
    "crime": "Axon",
    "police": "Axon",
    "tech": "Apple",
}

# Representative company -> ticker (Alpaca-compatible)
COMPANY_TICKER = {
    "Chevron": "CVX",
    "Exxon Mobil": "XOM",
    "Lockheed Martin": "LMT",
    "Boeing": "BA",
    "Fox Corporation": "FOX",
    "Alphabet": "GOOGL",
    "Xylem": "XYL",
    "AECOM": "ACM",
    "Caterpillar": "CAT",
    "Palantir": "PLTR",
    "Axon": "AXON",
    "Apple": "AAPL",
}

def extract_company_and_sentiment(text: str) -> dict:
    t = text.lower()
    company = None
    for keyword, comp in TOPIC_COMPANY.items():
        if re.search(rf"\b{re.escape(keyword)}\b", t):
            company = comp
            break

    score = float(analyzer.polarity_scores(text)["compound"])

    if company is None:
        return {"ticker": None, "score": 0.0}

    ticker = COMPANY_TICKER.get(company)
    if not ticker:
        return {"ticker": None, "score": 0.0}

    return {"ticker": ticker, "score": score}


# ----------------------------
# Simple BST for sentiment ranking
# ----------------------------
class Node:
    def __init__(self, score: float, ticker: str, count: int, day: str):
        self.score = float(score)
        self.ticker = ticker
        self.count = int(count)
        self.day = day  # YYYY-MM-DD
        self.left = None
        self.right = None


def bst_insert(root: Node | None, node: Node) -> Node:
    if root is None:
        return node
    if node.score < root.score:
        root.left = bst_insert(root.left, node)
    else:
        root.right = bst_insert(root.right, node)
    return root


def bst_leftmost(root: Node | None) -> Node | None:
    cur = root
    if cur is None:
        return None
    while cur.left is not None:
        cur = cur.left
    return cur


def bst_rightmost(root: Node | None) -> Node | None:
    cur = root
    if cur is None:
        return None
    while cur.right is not None:
        cur = cur.right
    return cur


# ----------------------------
# Persistent tree helpers
# ----------------------------
TREE_PATH = "sentiment_tree.pkl"


def load_tree(path: str) -> Node | None:
    if not os.path.exists(path):
        return None
    try:
        with open(path, "rb") as f:
            return pickle.load(f)
    except Exception as e:
        print(f"Warning: could not load existing tree from {path}: {e}")
        return None


def save_tree(path: str, root: Node | None) -> None:
    try:
        with open(path, "wb") as f:
            pickle.dump(root, f)
    except Exception as e:
        print(f"Warning: could not save tree to {path}: {e}")


# ----------------------------
# Scrape Truth Social posts
# ----------------------------
# Create a scraper session that handles Cloudflare
scraper = cloudscraper.create_scraper()

base_url = (
    f"https://truthsocial.com/api/v1/accounts/{USER_ID}/statuses"
    f"?exclude_replies={EXCLUDE_REPLIES}&only_replies=false&with_muted=true&limit={LIMIT}"
)

# ----------------------------
# Fetch posts with pagination (max_id)
# ----------------------------
max_id = None
all_posts = []

while len(all_posts) < LIMIT:
    paged_url = base_url
    if max_id:
        paged_url += f"&max_id={max_id}"

    posts = None
    last_resp = None

    for attempt in range(6):
        try:
            last_resp = scraper.get(paged_url, timeout=30)
        except Exception as e:
            wait = min(60, 2 ** attempt)
            print(f"Request error (attempt {attempt+1}/6): {e} | waiting {wait}s")
            time.sleep(wait)
            continue

        status = last_resp.status_code
        ctype = (last_resp.headers.get("content-type") or "").lower()

        if status == 429:
            ra = last_resp.headers.get("retry-after")
            try:
                wait = float(ra) if ra else RATE_LIMIT_FLOOR_SECONDS
            except ValueError:
                wait = RATE_LIMIT_FLOOR_SECONDS

            wait = max(wait, 2 ** attempt)
            wait = min(120.0, max(5.0, wait))
            print(f"Rate limited (HTTP 429). Retry-After={ra!r}. Waiting {wait:.1f}s")
            time.sleep(wait)
            continue

        if status in (500, 502, 503, 504):
            wait = min(60.0, 2 ** attempt)
            print(f"Truth Social HTTP {status} (attempt {attempt+1}/6) | waiting {wait:.1f}s")
            time.sleep(wait)
            continue

        try:
            posts = last_resp.json()
            time.sleep(max(0.0, PAGE_DELAY_SECONDS))
            break
        except ValueError:
            head = (last_resp.text or "")[:300]
            print(f"Non-JSON response. HTTP {status} | content-type: {ctype}")
            print(f"Response head: {head!r}")
            wait = min(60, 2 ** attempt)
            time.sleep(wait)
            continue

    if posts is None:
        break

    if isinstance(posts, dict) and "data" in posts:
        batch = posts["data"]
    elif isinstance(posts, list):
        batch = posts
    else:
        break

    if not batch:
        break

    all_posts.extend(batch)

    # prepare next page
    max_id = batch[-1].get("id")
    if not max_id:
        break

# all_posts now contains paginated results


# ----------------------------
# Extract text + run company/sentiment
# ----------------------------

processed = 0
trade_rows = []  # list of (ticker, score)

for post in all_posts:
    content_html = post.get("content", "")
    content_text = BeautifulSoup(content_html, "html.parser").get_text(strip=True)

    if not content_text or len(content_text) <= 3:
        continue

    text_for_model = normalize_post(content_text)
    if text_for_model is None:
        continue

    result = extract_company_and_sentiment(text_for_model)

    # Skip rows with no ticker (non-tradable / non-data)
    if result["ticker"] is None:
        continue

    trade_rows.append((result["ticker"], result["score"]))
    processed += 1

    if processed >= TARGET_USABLE_POSTS:
        break

print(f"Collected {processed} tradable signals for today.")

# ----------------------------
# Aggregate per ticker (mean sentiment)
# ----------------------------
agg_sum = defaultdict(float)
agg_count = defaultdict(int)
for tkr, sc in trade_rows:
    agg_sum[tkr] += float(sc)
    agg_count[tkr] += 1

aggregated = []  # list of (ticker, mean_score, count)
for tkr in agg_sum:
    mean_score = agg_sum[tkr] / max(1, agg_count[tkr])
    aggregated.append((tkr, mean_score, agg_count[tkr]))

# Build TODAY'S BST for today's picks
root_today = None
today_str = date.today().isoformat()
for tkr, mean_score, cnt in aggregated:
    root_today = bst_insert(root_today, Node(mean_score, tkr, cnt, today_str))

worst = bst_leftmost(root_today)
best = bst_rightmost(root_today)

if worst and best:
    print(
        f"Daily picks (today-only): SELL/SHORT {worst.ticker} (mean={worst.score:.4f}, n={worst.count}) | "
        f"BUY {best.ticker} (mean={best.score:.4f}, n={best.count})"
    )
elif best:
    print(f"Daily pick (today-only): BUY {best.ticker} (mean={best.score:.4f}, n={best.count})")
else:
    print("No tradable tickers found for daily picks today.")

# ----------------------------
# Update persistent BST (month-long history)
# ----------------------------
root_persist = load_tree(TREE_PATH)

inserted = 0
for tkr, mean_score, cnt in aggregated:
    root_persist = bst_insert(root_persist, Node(mean_score, tkr, cnt, today_str))
    inserted += 1

save_tree(TREE_PATH, root_persist)
print(f"Updated persistent tree: inserted {inserted} nodes for {today_str}. Saved to {TREE_PATH}.")