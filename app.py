from flask import Flask, render_template, url_for, redirect
from flask_bootstrap import Bootstrap

app = Flask(__name__)
Bootstrap(app)

@app.route('/',methods=['GET'])
def home():
    return render_template("index.html")

@app.route('/contact', methods=['GET'])
def contact():
    return render_template("base.html")

if __name__ == '__main__':
    app.run(host="0.0.0.0", port=8080, debug=True)