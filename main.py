from flask import Flask, render_template, request
import random
import string
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import nltk
import warnings

app = Flask(__name__)

warnings.filterwarnings('ignore')

nltk.download('punkt', quiet=True)
nltk.download('wordnet', quiet=True)

myfile = open("covid-19").read()

text = myfile
sent_tokens = nltk.sent_tokenize(text)

print(sent_tokens)

remove_punct_dict = dict((ord(punct), None) for punct in string.punctuation)

print(string.punctuation)
print(remove_punct_dict)


def LemNormalize(text):
    return nltk.word_tokenize(text.lower().translate(remove_punct_dict))


print(LemNormalize(text))

GREETING_INPUTS = ["hi", "hai", "hello", "hola", "greetings", "wassup", "hey"]
GREETING_RESPONSES = ["howdy", "hi", "hey", "what's good", "hello", "hey there"]


@app.route("/")
def home():
    return render_template("index.html")


def greeting(sentence):
    for word in sentence.split():
        if word.lower() in GREETING_INPUTS:
            return random.choice(GREETING_RESPONSES)


def response(user_response):
    user_response = user_response.lower()
    robo_response = ''
    sent_tokens.append(user_response)
    TfidfVec = TfidfVectorizer(tokenizer=LemNormalize, stop_words='english')
    tfidf = TfidfVec.fit_transform(sent_tokens)
    vals = cosine_similarity(tfidf[-1], tfidf)
    idx = vals.argsort()[0][-2]
    flat = vals.flatten()
    flat.sort()
    score = flat[-2]
    if (score == 0):
        robo_response = robo_response + "I apologize, I don't understand."
    else:
        robo_response = robo_response + sent_tokens[idx]
    sent_tokens.remove(user_response)
    return robo_response


@app.route("/get")
def get_response():
    user_response = request.args.get('msg')
    user_response = user_response.lower()
    print(user_response)
    return str(get_res(user_response))


def get_res(user_response):
    while 1:
        if (user_response != 'bye'):
            if (user_response == 'thanks' or user_response == 'thank you'):
                return str("Glad to help.")
            else:
                if (greeting(user_response) != None):
                    return str(greeting(user_response))
                else:
                    return str(response(user_response))
        else:
            return "Have a good day!"


if __name__ == "__main__":
    app.run(debug=True)
