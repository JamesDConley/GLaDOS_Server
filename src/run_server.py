# TODO : Install flask-session and redis in dockerfile
import zlib
import re
import time
import random
import traceback
import bleach
import logging
import pandoc
import redis
from flask import Flask, request, render_template, redirect, url_for, session
from flask_session import Session

# Local Imports
from clean_peft_model import GLaDOS
from compression import encode_str, decode_str, encode_obj, decode_obj

from waitress import serve
import queue


redis_host = "glados-redis"
redis_port = 6379
r = redis.Redis(host=redis_host, port=redis_port)

app = Flask(__name__)
app.config['SESSION_TYPE'] = 'redis'
app.config['SESSION_REDIS'] = r
app.secret_key = 'as89dvhuionasdjkfg'

Session(app)

LOG_FILE = "server_logs.log"
bot = GLaDOS("models/pythia_6b_r16_epoch1_54261/pytorch_model.bin", base_model_path="EleutherAI/pythia-6.9b-deduped", use_deepspeed=False, half=True, int8=False)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def commonmark_to_html(md_text):
    pd_data = pandoc.read(md_text, format="gfm")
    html_data = pandoc.write(pd_data, format="html")
    html_data = re.sub("<code[^>]*>", '<code class="prettyprint">', html_data)
    
    return html_data


def write_log(text, log_file=LOG_FILE):
    with open(log_file, "a") as fh:
        fh.write(text)

@app.route('/')
def splash_page():
    return render_template('form.html')

@app.route('/', methods=['POST'])
def start_convo():
    text = request.form.get('text', None)
    logger.debug(f"Received text : {text}")

    text = bleach.clean(text)
    session["new_text"] = encode_str(text)

    if request.form.get("process") == "Compare":
        return redirect(url_for("compare_results"))
    
    elif request.form.get("process") == "Submit":
        return redirect(url_for("conversation"))

def htmlify_convo(convo, speakers=("User", "GLaDOS")):
    md_messages = []
    for idx, message in enumerate(convo):
        #message = message.replace("\n", "<br>")
        d = {"speaker" : speakers[idx % 2], "html" : commonmark_to_html(message)}
        md_messages.append(d)
    return md_messages

def basicify_convo(convo,  speakers=("User", "GLaDOS")):
    md_messages = []
    for idx, message in enumerate(convo):
        message = message.replace("\n", "<br>")
        d = {"speaker" : speakers[idx % 2], "html" : f'<code>{message}</code>'}
        md_messages.append(d)
    return md_messages

@app.route("/toggle_md", methods=["GET", "POST"])
def toggle_md():
    original_val = session.get("use_md", True)
    session["use_md"] = not(original_val)
    return redirect("/")

@app.route('/conversation', methods=["GET", "POST"])
def conversation():
    # Pull text for inference
    new_text = session.get("new_text", None)
    session["new_text"] = None
    if new_text is not None:
        new_text = decode_str(new_text)
    if session.get("id", None) is None:
        session["id"] = random.randint(0, 9E4)
    if new_text is None:
        new_text = request.form.get('text', None)
        if new_text is None:
            # Error case
            return str(request.data.decode("utf-8"))
    if new_text is None:
        return "ERROR : Session has no new text"

    previous_convo = session.get("previous_convo", None)
    if previous_convo is not None:
        previous_convo = decode_obj(previous_convo)
    
    args = {
        "user_input" : new_text, 
        "conversation_history" : previous_convo, 
        "kwargs" : {"max_new_tokens":512, "do_sample":True, "temperature":1.2, "num_beams":2, "no_repeat_ngram_size" : 12, "top_k" : 50}
    }
    
    bot_response = bot.converse(**args)

    # Log stuff
    write_log(f"{time.time()} : {request.remote_addr} : Received request with text `{new_text}`\n")
    write_log(f"{time.time()} : {request.remote_addr} : Previous text was `{previous_convo}`\n")
    write_log(f"{time.time()} : {request.remote_addr} : Model response : `{bot_response}`\n")

    # Combine the full convo
    if previous_convo is None:
        previous_convo = []
    full_convo = previous_convo + [new_text, bot_response]

    # Save convo for next iteration
    session["previous_convo"] = encode_obj(full_convo)

    # Prepare convo for display
    use_md = session.get("use_md", True)
    if use_md:
        html_ready_convo = htmlify_convo(full_convo)
    else:
        html_ready_convo = basicify_convo(full_convo)

    return render_template("convo_2.html", messages=html_ready_convo)

@app.route('/clear')
def clear_user_posts():
    session["previous_convo"] = None
    return redirect("/")

@app.route("/options", methods=["GET", "POST"])
def compare_results():
    return "Not implemented, maybe go to http://jamesconley.net:5950/clear"

if __name__ == "__main__":
    # Dev
    #app.run(debug=True, port=5950)
    #serve(app, port=6000)

    # Prod
    print(f"HOORAY!!! STARTING APP!")
    serve(app, port=5950)
    # ~25 seconds to come online after running
