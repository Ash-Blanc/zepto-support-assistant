from flask import Flask, request, jsonify, render_template
from graph import graph

app = Flask(__name__)


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/ask", methods=["POST"])
def ask():

    data = request.get_json()

    question = data["question"]

    result = graph.invoke({
        "question": question,
        "intent": "",
        "answer": ""
    })

    return jsonify({
        "question": question,
        "answer": result["answer"]
    })


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)