from bottle import Bottle

from engine.app import app

web_server = Bottle()

@web_server.route('/')
def index():
    print(f'[WEB] GET /')
    return f'State: {app.current_state.name}'
