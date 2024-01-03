from flask import Flask, render_template, request , jsonify, session
from flask_session import Session
from flask_cors import CORS
import requests
import re
import json
import time
import pdfplumber

from PyPDF2 import PdfReader
from bs4 import BeautifulSoup
from collections import deque
from html.parser import HTMLParser
import os
import pandas as pd
import tiktoken
import openai
import numpy as np
from furl import furl
from openai.embeddings_utils import distances_from_embeddings, cosine_similarity
from ast import literal_eval
from selenium import webdriver
from dotenv import load_dotenv
from selenium.common import WebDriverException

load_dotenv()

app=Flask(__name__)
CORS(app)
app.config["SESSION_PERMANENT"] = False
app.config["SESSION_TYPE"] = "filesystem"
Session(app)

openai.api_key="openai_api_key"

@app.route("/")
def home():
    return "hello from home "




def create_context(chat_id, question, pre_conversation,  max_len=1800, size="ada"):
    # return  "context"
    main_df = pd.read_csv('processed/' + chat_id + '/embeddings.csv', index_col=0)
    main_df['embeddings'] = main_df['embeddings'].apply(literal_eval).apply(np.array)

    # question_embeddings = openai.Embedding.create(input=question, engine='text-embedding-ada-002')['data'][0]['embedding']
    response = openai.Embedding.create(input=question, engine='text-embedding-ada-002')
    question_embeddings = response['data'][0]['embedding']
    main_df['distances'] = distances_from_embeddings(question_embeddings, main_df['embeddings'].values, distance_metric='cosine')

    returns = []
    cur_len = 0
    
    conversation_ = f"Input: {pre_conversation}\n\n"
    # conversation_ = "Input: {}\n\n".format(pre_conversation)

    for i, row in main_df.sort_values('distances', ascending=True).iterrows():
        cur_len += row['token'] + 4
        if cur_len > max_len:
            break
        returns.append(row["text"])

    return "\n\n###\n\n".join([conversation_] + returns)

# model="text-davinci-003",
def answer_question_with_memory(chat_id,user_question, conversation, model="gpt-3.5-turbo" , max_len=1800, size="ada",debug=False, max_tokens=150, stop_sequence=None):
    
    # return ""
    
    context = create_context(chat_id, user_question, conversation, max_len=max_len, size="ada")
    
    if debug:
        print("Context:\n" + context)
        print("\n\n")


    response = openai.ChatCompletion.create(
        model='gpt-3.5-turbo',
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": f"Answer the question based on the context below, and if the question can't be answered based on the context, say \"I don't know\"\n\nContext: {context}\n\n---\n\nQuestion: {user_question}\nAnswer:"}
        ],
        temperature=0,
        max_tokens=max_tokens,
        top_p=1,
        frequency_penalty=0,
        presence_penalty=0,
        stop=stop_sequence,
    )

    return response["choices"][0]["message"]["content"].strip()

    


@app.route("/api/ask", methods=["POST"])
def ask():
    
    chat_id="qwerty"
    # chat_id = request.args.get('chat_id')
    data = request.json
    user_input = data.get('user_input', '')
    
    if 'conversation_history' not in session:
        session['conversation_history'] = []

    bot_response = answer_question_with_memory(chat_id, user_input, session['conversation_history'], max_tokens=150)
    session['conversation_history'].append((user_input, bot_response))

    return jsonify(
        {"answer": bot_response}
    )






# PDF PROCESS
def get_raw_text_from_pdf(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
            print(text)
    return text


def process_raw_text_and_save(text):
    
    chat_id = "qwerty"

    if not os.path.exists("text/"):
        os.mkdir("text/")

    if not os.path.exists("text/" + chat_id + "/"):
        os.mkdir("text/" + chat_id + "/")

    if not os.path.exists("processed/"):
        os.mkdir("processed/")

    if not os.path.exists("processed/" + chat_id + "/"):
        os.mkdir("processed/" + chat_id + "/")

    tokenizer = tiktoken.get_encoding("cl100k_base")

    max_tokens_ = 500
    shortened = []

    if len(tokenizer.encode(text)) > max_tokens_:
        shortened += split_into_chunks(text)
    else:
        shortened.append(text)
    
    file_scraped="ChunksWithTokenCount.csv"
    
    df = pd.DataFrame(shortened, columns=['text'])
   
    df['token'] = df.text.apply(lambda x: len(tokenizer.encode(x)))
    df.to_csv('processed/' + chat_id + '/' + file_scraped)
    # Call to open AI API for embeddings
    df['embeddings'] = df.text.apply(
        lambda x: openai.Embedding.create(input=x, engine='text-embedding-ada-002')['data'][0]['embedding'])
    df.to_csv("processed/" + chat_id + "/embeddings.csv")

    return ""
# PDF PROCESS



# URL PROCESS
class HyperlinkParser(HTMLParser):

    def __init__(self):
        super().__init__()
        # Create a list to store the hyperlinks
        self.hyperlinks = []

    # Override the HTMLParser's handle_starttag method to get the hyperlinks
    def handle_starttag(self, tag, attrs):
        attrs = dict(attrs)

        # If the tag is an anchor tag, and it has a href attribute, add the href attribute to the list of hyperlinks
        if tag == "a" and "href" in attrs:
            self.hyperlinks.append(attrs["href"])


def get_hyperlinks(url):
    try:
        response = requests.get(url)
        response.raise_for_status()  # Raise an exception for HTTP errors

        if not response.headers.get('Content-Type', '').startswith("text/html"):
            return []

        html = response.text
    except requests.exceptions.RequestException as e:
        print(e)
        return []

    parser = HyperlinkParser()
    parser.feed(html)

    return parser.hyperlinks


def get_domain_hyperlinks(local_domain, url):
    # Regex pattern to match a URL
    HTTP_URL_PATTERN = r'^http[s]{0,1}://.+$'

    clean_links = []
    for link in set(get_hyperlinks(url)):
        clean_link = None

        # If the link is a URL, check if it is within the same domain
        if re.search(HTTP_URL_PATTERN, link):
            url_obj = furl(link)

            if url_obj.netloc == local_domain:
                clean_link = link

        # If the link is not a URL, check if it is a relative link
        else:
            if link.startswith("/"):
                link = link[1:]
            elif (
                    link.startswith("#")
                    or link.startswith("mailto:")
                    or link.startswith("tel:")
            ):
                continue
            clean_link = "https://" + local_domain + "/" + link

        if clean_link is not None:
            if clean_link.endswith("/"):
                clean_link = clean_link[:-1]
            clean_links.append(clean_link)

    # Return the list of hyperlinks that are within the same domain
    return list(set(clean_links))


def split_into_chunks(text):
    tokenizer = tiktoken.get_encoding("cl100k_base")
    max_tokens = 500
    # Split the text into sentences
    sentences = text.split('. ')

    # Get the number of tokens for each sentence
    n_tokens = [len(tokenizer.encode(" " + sentence)) for sentence in sentences]

    chunks = []
    tokens_so_far = 0
    chunk = []

    # Loop through the sentences and tokens joined together in a tuple
    for sentence, token_count in zip(sentences, n_tokens):
        # If the number of tokens so far plus the number of tokens in the current sentence is greater
        # than the max number of tokens, then add the chunk to the list of chunks and reset
        # the chunk and tokens so far
        if tokens_so_far + token_count > max_tokens:
            chunks.append(". ".join(chunk) + ".")
            chunk = []
            tokens_so_far = 0
        # If the number of tokens in the current sentence is greater than the max number of
        # tokens, go to the next sentence
        if token_count > max_tokens:
            continue
        # Otherwise, add the sentence to the chunk and add the number of tokens to the total
        chunk.append(sentence)
        tokens_so_far += token_count + 1

    # Add the last chunk to the list of chunks
    if chunk:
        chunks.append(". ".join(chunk) + ".")

    return chunks


def remove_newlines(serie):
    serie = serie.str.replace('\n', ' ')
    serie = serie.str.replace('\\n', ' ')
    serie = serie.str.replace('  ', ' ')
    serie = serie.str.replace('  ', ' ')
    return serie


def process_text_files(loc_domain):
   
    global df
   
    texts = []

    for file in os.listdir("text/" + loc_domain + "/"):
        with open("text/" + loc_domain + "/" + file, "r", encoding="UTF-8") as f:
            text = f.read()
            texts.append((file[11:-4].replace('-', ' ').replace('_', ' ').replace('#update', ''), text))

    file_scraped = "scraped.csv"

    df = pd.DataFrame(texts, columns=['title', 'text'])
    df['text'] = df.title + ". " + remove_newlines(df.text)
    df.to_csv('processed/' + loc_domain + '/' + file_scraped)

    tokenizer = tiktoken.get_encoding("cl100k_base")
    df = pd.read_csv('processed/' + loc_domain + '/' + file_scraped, index_col=0)
    df['token'] = df.text.apply(lambda x: len(tokenizer.encode(x)))
   


    max_tokens_ = 500
    shortened = []

    for row in df.iterrows():
        if row[1]['text'] is None:
            continue
        if row[1]['token'] > max_tokens_:
            shortened += split_into_chunks(row[1]['text'])
        else:
            shortened.append(row[1]['text'])

    df = pd.DataFrame(shortened, columns=['text'])
    df['token'] = df.text.apply(lambda x: len(tokenizer.encode(x)))
    df.to_csv('processed/' + loc_domain + '/' + file_scraped)
    
    # Call to open AI api
    df['embeddings'] = df.text.apply(lambda x: openai.Embedding.create(input=x, engine='text-embedding-ada-002')['data'][0]['embedding'])
    df.to_csv("processed/" + loc_domain + "/embeddings.csv")

    # Read the value from embeddings.csv and use it to answer the question
    # df = pd.read_csv('processed/' + loc_domain + '/embeddings.csv', index_col=0)
    # df['embeddings'] = df['embeddings'].apply(literal_eval).apply(np.array)


def crawl(url):

    local_domain = furl(url).host
    queue = deque([url])
    seen = {url}

    if not os.path.exists("text/"):
        os.mkdir("text/")

    if not os.path.exists("text/" + local_domain + "/"):
        os.mkdir("text/" + local_domain + "/")

    if not os.path.exists("processed/"):
        os.mkdir("processed/")

    if not os.path.exists("processed/" + local_domain + "/"):
        os.mkdir("processed/" + local_domain + "/")

    options = webdriver.ChromeOptions()
    options.add_argument("--headless")
    driver = webdriver.Chrome(options=options)

    max_retries = 3
    retry_delay = 2
    timeout = 10
    retries = 0

    while queue and retries < max_retries:
        url = queue.pop()
        try:
            driver.get(url)
        except WebDriverException as e:
            print(f"Error loading URL: {url}\nError: {e}")
            retries += 1
            time.sleep(retry_delay)
            continue  

        with open('text/' + local_domain + '/' + url[8:].replace("/", "_") + ".txt", "w", encoding="UTF-8") as f:
            soup = BeautifulSoup(driver.page_source, "html.parser")
            text = soup.get_text()
            f.write(text)

        for link in get_domain_hyperlinks(local_domain, url):
            if link not in seen:
                queue.append(link)
                seen.add(link)

    driver.quit()

    process_text_files(local_domain)
# URL PROCESS


@app.route("/api/savedata", methods=["POST"])
def save_data():

    option = request.form.get('option')

    if option == 'pdf':
        pdf_files = request.files.getlist("file") # Access the uploaded file
        if pdf_files:
            text_per_pdf = ""
            for pdf_file in pdf_files:
                with pdfplumber.open(pdf_file) as pdf:
                    for page in pdf.pages:
                        text_per_page = page.extract_text()
                        text_per_pdf += text_per_page  # Accumulate text for each page
            process_raw_text_and_save(text_per_pdf)
            num_files_uploaded = len(pdf_files)  # Get the count of uploaded files
            print(f"Number of PDF files uploaded: {num_files_uploaded}")
    
    elif option == 'url':
        url = request.form.get('url')
        crawl(url)

    return jsonify({"message": "Data saved successfully"})





@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        username = request.form.get("username")
        password = request.form.get("password")

        # Check if the user exists and the password is correct
        for user in users:
            if user["username"] == username and user["password"] == password:
                return f"Welcome, {username}!"
        return "Invalid credentials. Please try again."

    return "login-route"



@app.route("/register", methods=["GET", "POST"])
def register():
    if request.method == "POST":
        username = request.form.get("username")
        password = request.form.get("password")

        # Add the new user to the list
        users.append({"username": username, "password": password})
        return f"Registered successfully! You can now <a href='/login'>login</a>."

    return "register-route"



if __name__ == "__main__":
    app.run(debug=True)
