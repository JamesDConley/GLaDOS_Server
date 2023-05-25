# TODO : REFACTOR THIS FILE SO THAT IT USES A CLASS FOR FLASK
# RIGHT NOW EVERYTHING IS JUST LOOSE IN FILE AND DAS BAD MMKAY

# TODO : FIX THE ABOVE AS SOON AS THE STREAMING FEATURE IS WORKING
import time
import random
import logging

import bleach
import redis
from flask import Flask, request, render_template, redirect, url_for, session
from flask_session import Session
from waitress import serve
from flask_sse import sse

# Local Imports
from glados import GLaDOS
from md_utils import htmlify_convo, basicify_convo, commonmark_to_html
from get_args import get_args

# Conversations are stored compressed
from compression import encode_obj, decode_obj
from streamer import CallbackStreamer

# Get Args
args = get_args()

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
app.config["REDIS_URL"] = "redis://glados-redis"
# TODO : This probably needs to changed in prod for security or something
app.secret_key = 'as89dvhuionasdjkfg'

# Make the app use redis for sessions
Session(app)
app.register_blueprint(sse, url_prefix="/stream")
LOG_FILE = "server_logs.log"

# Testing model
bot = GLaDOS("JamesConley/glados_together_20b_lora_merged", token=args.token, multi_gpu=True)
#bot = GLaDOS("unionai/pythia-410m-finetune-alpaca", token=args.token, multi_gpu=args.multi_gpu)
#bot = GLaDOS(args.model, token=args.token, multi_gpu=args.multi_gpu)
bot.add_stop_phrase("User:")

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
    session["new_text"] = text

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

@app.route("/generate_text")
def generate_text():
    time.sleep(0.5)
    new_text = session.get("new_text", None)

    # Make sure the session has an ID
    if session.get("id", None) is None:
        session["id"] = random.randint(0, 9E4)
    
    # Log potential error case
    if new_text is None:
        return "ERROR : Session has no new text"
    
    previous_convo = session.get("previous_convo", None)
    if previous_convo is not None:
        previous_convo = decode_obj(previous_convo)
    
    
    prompt = bot.build_prompt(new_text, previous_convo)
    use_md = session.get("use_md", True)
    def my_callback(tokens):
        partial_text = bot.decode_token_seq(tokens, truncate=False)
        print(partial_text)
        if not use_md:
            partial_text = commonmark_to_html(partial_text)
        sse.publish({"text": partial_text[len(prompt):]}, type="text_update")
    cbs = CallbackStreamer(my_callback, my_callback)
    args = {
        "user_input" : new_text,
        "conversation_history" : previous_convo, 
        "kwargs" : {"max_new_tokens":512, "do_sample":False, "temperature":1.0, "num_beams":1, "no_repeat_ngram_size" : 5, "top_k" : 50, "streamer" : cbs}
    }
    bot_response = bot.converse(**args)

    # Log stuff
    logger.debug(f"{time.time()} : {request.remote_addr} : Received request with text `{new_text}`\n")
    logger.debug(f"{time.time()} : {request.remote_addr} : Previous text was `{previous_convo}`\n")
    logger.debug(f"{time.time()} : {request.remote_addr} : Model response : `{bot_response}`\n")
    ### START OF PASTED CODE ###
    
    # Combine the full convo
    if previous_convo is None:
        previous_convo = []
    
    full_convo = previous_convo + [new_text, bot_response]
    # Save convo for next iteration
    session["previous_convo"] = encode_obj(full_convo)
    session["new_text"] = None
    return "Text generation completed!"

@app.route('/conversation', methods=["GET", "POST"])
def conversation():
    """Render a conversation with the bot"""
    new_text = session.get("new_text", None)
    if new_text is None:
        new_text = request.form.get('text', None)
        session["new_text"] = new_text
    previous_convo = session.get("previous_convo", None)
    if previous_convo is not None:
        previous_convo = decode_obj(previous_convo)

    # Combine the full convo
    if previous_convo is None:
        previous_convo = []
    previous_convo = previous_convo + [new_text]
    print(f"Previous Convo is : {previous_convo}")
    # Prepare convo for display
    use_md = session.get("use_md", True)
    if use_md:
        html_ready_convo = htmlify_convo(previous_convo)
    else:
        html_ready_convo = basicify_convo(previous_convo)

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
    print("STARTING APP!!!")
    serve(app, port=5950)
