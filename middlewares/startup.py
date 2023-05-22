from flask import Flask
from flask_pymongo import PyMongo

class StartUp:
  def db(app):
    app.config["MONGO_URI"] = "mongodb://localhost:27017/ThreePhase"
    mongo = PyMongo(app)
    return mongo
