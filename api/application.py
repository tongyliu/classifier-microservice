import db
from flask import Flask

application = Flask(__name__)


@application.route('/health', methods=['GET'])
def health():
    conn = db.connect()
    print(conn)
    return {'message': 'Ok'}

