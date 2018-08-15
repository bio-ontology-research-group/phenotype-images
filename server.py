from flask import Flask, jsonify, request
from flask_cors import CORS, cross_origin
from ExtraFunctions import find, firstFilter, Search
app = Flask(__name__)

app.config['SECRET_KEY'] = 'annoying'
app.config['CORS_HEADERS'] = 'Content-Type'

@app.route("/load/")
@cross_origin(origin='localhost', headers=['Content- Type', 'Authorization'])
def initial_filer():
    return jsonify(firstFilter("10kimages.txt"))

@app.route("/metadata/<string:data_id>")
@cross_origin(origin='localhost', headers=['Content- Type', 'Authorization'])
def get_person_by_id(data_id):
    return jsonify(find(data_id, "10kimages.txt"))


@app.route('/search/', methods=['POST'])
@cross_origin(origin='localhost', headers=['Content- Type', 'Authorization'])
def worker():
    data = request.form['json']
    Search("10kimages.txt", "L1000.csv", data)

    return "bye"

app.run(port=19000)

