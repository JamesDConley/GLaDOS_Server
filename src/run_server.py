import re
import time
import random
import logging

import bleach
import pandoc
import redis
from flask import Flask, request, render_template, redirect, url_for, session
from flask_session import Session
from waitress import serve

# Local Imports
from glados import GLaDOS
from md_utils import htmlify_convo basicify_convo

# Conversations are stored compressed
from compression import encode_str, decode_str, encode_obj, decode_obj

# Setup Logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Setup redis connection (stores conversations)
redis_host = "glados-redis"
redis_port = 6379
r = redis.Redis(host=redis_host, port=redis_port)

# Setup Flask app
app = Flask(__name__)
app.config['SESSION_TYPE'] = 'redis'
app.config['SESSION_REDIS'] = r
# TODO : This probably needs to changed in prod for security or something
app.secret_key = 'as89dvhuionasdjkfg'

# Make the app use redis for sessions
Session(app)

LOG_FILE = "server_logs.log"
bot = GLaDOS("models/GLaDOS20B", use_deepspeed=False, half=True, int8=False, multi_gpu=False)

@app.route('/')
def splash_page():
    # Main page for the website
    return render_template('form.html')

@app.route('/', methods=['POST'])
def start_convo():
    # Capture form posts and redirect them to the appropriate page
    text = request.form.get('text', None)
    logger.debug(f"Received text : {text}")

    text = bleach.clean(text)
    session["new_text"] = encode_str(text)

    # TODO : Create compare results page for sampling/RLHF examples
    if request.form.get("process") == "Compare":
        return redirect(url_for("compare_results"))
    
    elif request.form.get("process") == "Submit":
        return redirect(url_for("conversation"))

@app.route("/toggle_md", methods=["GET", "POST"])
def toggle_md():
    # Toggle between rendered HTML or plain markdown outputs in the UI
    original_val = session.get("use_md", True)
    session["use_md"] = not(original_val)
    return redirect("/")

@app.route('/conversation', methods=["GET", "POST"])
def conversation():
    # Render a conversation with the bot
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
        "kwargs" : {"max_new_tokens":512, "do_sample":True, "temperature":1.0, "num_beams":1, "no_repeat_ngram_size" : 12, "top_k" : 50}
    }
    try:
        bot_response = bot.converse(**args)
    # TODO : Bare exception. What is the error when it OOMs? Is it reliable?
    except:
        bot_response = f"ERROR : Bot code errored- likely OOM from longer conversation. Consider going to http://jamesconley.net:5950/clear or clicking the back button to reset the thread"
    
    # Log stuff
    logger.debug(f"{time.time()} : {request.remote_addr} : Received request with text `{new_text}`\n")
    logger.debug(f"{time.time()} : {request.remote_addr} : Previous text was `{previous_convo}`\n")
    logger.debug(f"{time.time()} : {request.remote_addr} : Model response : `{bot_response}`\n")

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

    return render_template("conversation_page.html", messages=html_ready_convo)

@app.route('/clear')
def clear_user_posts():
    session["previous_convo"] = None
    return redirect("/")

@app.route("/options", methods=["GET", "POST"])
def compare_results():
    return "Not implemented, maybe go to [base_url]/clear"

if __name__ == "__main__":
    # Prod
    print(f"STARTING APP!!!")
    serve(app, port=5950)
