from flask import Flask, render_template, request, session

import json
import requests
import logging
import random

app = Flask(__name__)
app.config['SECRET_KEY'] = 'aydaydyad'
app.config['FORMALITY_API_URL'] = 'http://formality_api:5000'

def get_random_text(idx=None):
    texts = [
        """
    Hi mate!
This app let's you test your formal writing skill. 
Just type your formal email or a post here and click analyze.
    Don't know what to type? Don't worry, just click example and see much it scores.
Under the formality score you can see your text highlighted with yellow color.
Notice how different words are highlighted differently. 
The more intense the color, the more informal the highlighted word sounds in this context.
Try changing the word or rewrite the sentence to improve the score.
        """,
        
        """
Compensation is, after all, supposed to bear some relationship - a close relationship, in a competitive market - to the actual skills that individuals possess and the manner in which they can deliver value to firm clients
Tying that compensation rigidly to associate seniority makes about as much sense for law firm associates as it does for elementary school teachers, i.e., none at all.
        """,

        """
Adipic acid is primarily used in the production of nylon 6-6 and for manufacturing polyurethane foam and polyester resins.
        """,
        """
Among the thirteen new taxes found in the bill by Americans for Tax Reform is the ""medicine cabinet tax,"" which bars people from paying for non-prescription medicine with tax-deferred health savings accounts.
        """,
        """
And BTW, if I add Scribus to the default programs list it will appear twice in the contextual menu and it will appear in the menu of plain text files, which I don't want to.
        """,
        """
Cousin Cathy was a classy little lady who adored minuets and Crepe Suzette; rock 'n' roller cousin Patty lost control when it came to, um, hot dogs.
        """,

        """
I guess those kids thought what they did was tight, was cool. But it was terrible.
        """
    ]
    if idx is None:
        idx = random.randint(0, len(texts) - 1)
    return texts[idx]

@app.route('/', methods=['GET', 'POST'])
def index():
    
    if request.method == 'POST':
        text = request.form['text']
        formality_request = requests.post(app.config['FORMALITY_API_URL'] + '/predict', json={"text": text})
        response = json.loads(formality_request.text)
        formality_score = response["score"]
        word_scores = response["word_scores"]
        logging.info(f"Got response: {response}")

        res = {}
        res['text'] = text
        res['formality_score'] = int(100 * formality_score)
        logging.debug(f"Total formality score = {formality_score}")
        res['text_list'] = []
        res['score_list'] = []
        punct_symbols = ".,:;!?'"
        for sentence in word_scores:
            sentence_informality_score = sentence["informality_score"]
            sentence_word_scores = sentence["word_scores"]
            for (word, word_score) in sentence_word_scores:
                if word in punct_symbols: # append punctuation symbol to the last word
                    res['text_list'][-1] += word
                    continue
                if word.startswith("##"): # merge subword tokens with the last word
                    res['text_list'][-1] += word[2:]
                    res['score_list'][-1] += word_score
                    continue
                res['text_list'].append(" " + word)
                word_score *= sentence_informality_score
                res['score_list'].append(word_score)

        res['words_scored'] = list(zip(res['text_list'], res['score_list']))
        results = None
        return render_template('index.html', res=res)

    if 'index' in session:
        results = session['index']
        return render_template('index.html', res=results)

    res = {}
    res["text"] = get_random_text(0)
    return render_template('index.html', res=res)

@app.route('/example', methods=['GET', 'POST'])
def example():
    res = {}
    res["text"] = get_random_text()
    return render_template('index.html', res=res)

if __name__ == "__main__":
    logging.basicConfig(encoding='utf-8', level=logging.DEBUG)
    from os import environ
    app.run(debug=False, host='0.0.0.0', port=environ.get("PORT", 5050))