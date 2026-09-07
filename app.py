# Importing the required libraries
import spacy
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
from collections import Counter
from flask import Flask, render_template, request, url_for

# Downloading the Vader lexicon.
nltk.download('vader_lexicon')

# Initialising the NLP models and Flask
ner = spacy.load("en_core_web_trf")
sia = SentimentIntensityAnalyzer()
app = Flask(__name__)

# 1. Route that loads the home page
@app.route("/")
def home():
    return render_template('index.html')

# 2. Route that handles text submissions, runs the NLP analysis, and returns results
@app.route('/analyze', methods=['POST'])
def analyze():
    submitted_text = request.form.get('manuscript_text')
    print('Success! Received text:')

    entities = ner(submitted_text)

    # Filtering entities to show only the most relevant categories
    relevant_labels = ["PERSON", "LOC", "GPE", "ORG"]
    extracted_entities = [
        (ent.text, ent.label_)
        for ent in entities.ents
        if ent.label_ in relevant_labels
        ]
    entity_frequencies = Counter(extracted_entities)

    # saving the results of the entity recogniser in a dictionary
    ner_records = {}
    for (entity, lab), count in entity_frequencies.most_common():

        # renaming labels to make them easier to read
        if lab == 'LOC':
            label = 'Location'
        elif lab == 'PERSON':
            label = 'Person'
        elif lab == 'GPE':
            label = 'Geopolitical Entity'
        elif lab == 'ORG':
            label = 'Organisation'
        else:
            label = lab
        ner_records[entity] = {
            'label': label,
            'count': count
        }

    # saving the results of the sentiment analyser for each sentence in a dictionary
    sentences_records = {}
    for sent in entities.sents:
        sentences_records[sent.text] = sia.polarity_scores(sent.text)

    # saving the results of the sentiment analyser for the entire text submission
    overall_sentiment = sia.polarity_scores(submitted_text)

    # the function will return the results html
    return render_template('results.html', 
                           entities=ner_records, 
                           sentences=sentences_records, 
                           overall_sentiment=overall_sentiment)

# Debugger mode enabled to apply real time developments
if __name__ == '__main__':
    app.run(debug=True)



