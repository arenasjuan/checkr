import os

# Suppress TensorFlow logging (1 = INFO messages are not printed)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'

import re
import requests
import praw
import config
import spacy
import openai
from datetime import datetime
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


# For using spaCy with GPU:
spacy.prefer_gpu()
nlp = spacy.load("en_core_web_sm")


client = openai.OpenAI(api_key = config.openai_api_key)

def get_comment_text(permalink):
    # Fetch the comment
    comment = reddit.comment(url=permalink)

    # For accessing comment creation date:
    # comment_creation_date = datetime.utcfromtimestamp(comment.created_utc)
    # print(f"\n\nComment created: {comment_creation_date}\n\n")

    return comment.body

def calculate_cosine_similarity(text1, text2):
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform([text1, text2])
    return cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0]

def nyt_api_call(gpt_response):
    # Extract the comment abstract
    abstract_match = re.search(r'Comment Abstract: (.*)', gpt_response)
    comment_abstract = abstract_match.group(1) if abstract_match else None

    if not comment_abstract:
        print("No comment abstract found in the GPT-3.5 response.")
        return

    print(f"\nComment Abstract: {comment_abstract}\n")

    # Extract URL from the GPT response
    url_pattern = r'https://api\.nytimes\.com/svc/search/v2/articlesearch\.json\?q=[\w%+]+'
    match = re.search(url_pattern, gpt_response)

    if not match:
        print("No valid NYT API URL found in the GPT-3.5 response.")
        return

    api_url = match.group(0) + f"&api-key={config.nyt_api_key}"
    print(f"\nMaking NYT API Call to URL: {api_url}\n")
    response = requests.get(api_url)

    cosine_scores = []
    if response.status_code == 200:
        nyt_data = response.json()
        print(f"\nReceived Data: {nyt_data}\n")
        if 'docs' in nyt_data.get("response", {}):
            for article in nyt_data["response"]["docs"]:
                nyt_abstract = article.get("abstract", "")
                if nyt_abstract:
                    similarity_score = calculate_cosine_similarity(comment_abstract, nyt_abstract)
                    cosine_scores.append(similarity_score)
                    print(f"Article URL: {article.get('web_url')}, Similarity Score: {similarity_score}")
                else:
                    print("No abstract available in this article.")
        else:
            print("No articles found in the NYT response.")
    else:
        print(f"Error in NYT API call: {response.status_code}, {response.text}")

    return cosine_scores
    

permalink = config.example_comment

# Initialize PRAW
reddit = praw.Reddit(
    client_id = config.reddit_client_id,
    client_secret = config.reddit_client_secret,
    user_agent = 'checkr',
)


response = client.chat.completions.create(
    model="gpt-3.5-turbo",
    temperature=0.4,
    messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": f"Generate a New York Times Article Search API URL (which should begin 'https://api.nytimes.com/svc/search/v2/articlesearch.json?q=') with the 5 most important keywords related to the following text, then print an abstract of the claim(s) being made in the text (which should begin 'Comment Abstract: ': {get_comment_text(permalink)}"},
    ],
    #top_p = 1.0,
    #stream = False,
    #max_tokens = 100,
    #frequency_penalty = 0.0,
    #presence_penalty = 0.0,
)

nyt_api_call(response.choices[0].message.content)

# print(f"\n\nResponse check: {response}\n")