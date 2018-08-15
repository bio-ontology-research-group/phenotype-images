from flask import Flask, jsonify, request
from flask_cors import CORS, cross_origin
from ExtraFunctions import find, firstFilter, Search
app = Flask(__name__)

app.config['SECRET_KEY'] = 'annoying'
app.config['CORS_HEADERS'] = 'Content-Type'

@app.route("/load/")
@cross_origin(origin='localhost', headers=['Content- Type', 'Authorization'])
def initial_filer():
    return jsonify(firstFilter("json20k.txt"))

@app.route("/metadata/<string:data_id>")
@cross_origin(origin='localhost', headers=['Content- Type', 'Authorization'])
def get_person_by_id(data_id):
    return jsonify(find(data_id, "json20k.txt"))


@app.route('/search/', methods=['POST'])
@cross_origin(origin='localhost', headers=['Content- Type', 'Authorization'])
def worker():
    data = request.form['json']
    Search("json20k.txt", "L20000.csv", data)

    return "bye"

app.run(port=19000)

