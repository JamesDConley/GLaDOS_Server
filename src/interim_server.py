
from flask import Flask, request, render_template, redirect, url_for, session
from waitress import serve


app = Flask(__name__)


@app.route('/')
def splash_page():
    return "James is currently using the GPU for other things. Thanks for checking it out though!"
if __name__ == "__main__":
    # Prod
    serve(app, port=5950)
