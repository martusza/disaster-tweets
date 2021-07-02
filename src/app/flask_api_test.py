from flask import Flask, make_response

app = Flask(__name__)


@app.route('/')
def index():
    return "Hello"


@app.route('/<param>')
def model(param):
    return f"HERE IS WHAT YOU TYPE: \n{param}"


@app.route("/test")
def users():
    return make_response(
        'Test worked!',
        200
    )


if __name__ == "__main__":
    app.run(host='0.0.0.0')

# localhost:5000
#  http://127.0.0.1:5000/
