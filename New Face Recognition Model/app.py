from flask import Flask, render_template
from flask_socketio import SocketIO, emit
import cv2

app = Flask(__name__)
socketio = SocketIO(app)

# Funksioni për të dërguar frames përmes WebSockets
def video_stream():
    cap = cv2.VideoCapture(0)  # Mund të zëvendësoni këtë me kamerën tuaj
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        # Konvertoni frame në jpeg
        _, jpeg = cv2.imencode('.jpg', frame)
        frame = jpeg.tobytes()
        # Dërgojeni frame përmes WebSocket
        socketio.emit('video_frame', frame)

@app.route('/')
def index():
    return render_template('index.html')

if __name__ == "__main__":
    socketio.start_background_task(target=video_stream)
    socketio.run(app)
