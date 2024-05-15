import os
from main import socketio, app

if __name__ == "__main__":
    #socketio.run(app, host='0.0.0.0', port=8080, debug=False)
    socketio.run(app, host="0.0.0.0", port=int(os.environ.get("PORT", 8080)), debug=False)
    #app.run(host='0.0.0.0', port=8080)
