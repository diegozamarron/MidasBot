import cloudscraper
from bs4 import BeautifulSoup
import csv

# Create a scraper session that handles Cloudflare
scraper = cloudscraper.create_scraper()

# Replace with the numeric Truth Social user ID
user_id = "107780257626128497"

base_url = f"https://truthsocial.com/api/v1/accounts/{user_id}/statuses?exclude_replies=true&only_replies=false&with_muted=true&limit=100"

all_posts = []
max_id = None

while True:
    url = base_url
    if max_id:
        url += f"&max_id={max_id}"

    response = scraper.get(url)
    try:
        posts = response.json()
    except ValueError:
        print("Response is not JSON. Likely blocked by Cloudflare or missing authentication.")
        break

    # Determine iterable
    if isinstance(posts, dict) and 'data' in posts:
        iterable = posts['data']
    elif isinstance(posts, list):
        iterable = posts
    else:
        print("Unexpected JSON structure")
        break

    if not iterable:
        break  # no more posts

    all_posts.extend(iterable)
    max_id = iterable[-1]['id']  # pagination

# Save only textual posts to CSV with separators
with open('truthsocial_posts_text.csv', 'w', newline='', encoding='utf-8') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(['Post'])

    for post in all_posts:
        content_html = post.get('content', '')
        content_text = BeautifulSoup(content_html, 'html.parser').get_text(strip=True)

        # Skip empty posts or posts that are very short
        if content_text and len(content_text) > 10:
            print(content_text)
            print('\n' + '-'*80 + '\n')
            writer.writerow([content_text])