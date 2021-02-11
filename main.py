from flask import Flask, render_template
import controller.CharacterController as characterController

app = Flask(__name__)

app.register_blueprint(characterController.characterBP, url_prefix='/character')


@app.route('/')
def index():
    return render_template('characterView.html')
