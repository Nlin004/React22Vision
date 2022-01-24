# Writing the Tornado server to initiate the client request

import tornado.ioloop
import tornado.web
import tornado.websocket
import cv2
from img_to_string import to_b64
from main import main, source

from tornado.options import define, options

define('port', default=8080, type=int)

cap = cv2.VideoCapture(source)
# cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0)
cap.set(cv2.CAP_PROP_EXPOSURE, -4)

# This handler handles a call to the base of the server \
# (127.0.0.1:8888/ -> 127.0.0.1:8888/index.html)
class IndexHandler(tornado.web.RequestHandler):
    # GET request to get the base webpage
    # from the Tornado server
    def get(self):
        self.render('websocket/www/index.html')

# This handler handles a websocket connection
class WebSocketHandler(tornado.websocket.WebSocketHandler):
    # function to open a new connection to the WebSocket
    def open(self, *args):
        print('new connection!')
        # self.write_message('welcome!')

    # function to respond to a message on the WebSocket
    def on_message(self, message):
        _, frame = cap.read()
        
        # enter open cv code here
        output_image = main(frame, message)
        cv2.imwrite("./websocket/frame.jpg", output_image)

        self.write_message(to_b64("./websocket/frame.jpg"))

    # function to close a connection on the WebSocket
    def on_close(self):
        print('connection closed')

class CameraHandler(tornado.web.RequestHandler):
    def get(self):
        self.render("./www/camera.html")

class CameraSocketHandler(tornado.websocket.WebSocketHandler):
    def open(self, *args):
        print("camera websocket connection")
    
    def on_message(self, message):
        self.write_message(message)

    def on_close(self):
        print('connection closed')

app = tornado.web.Application([
    (r'/', IndexHandler),
    (r'/ws/', WebSocketHandler),
    (r'/camera/', CameraHandler),
    (r'/camera/ws/', CameraSocketHandler)
])

if __name__ == '__main__':
    app.listen(options.port)
    tornado.ioloop.IOLoop.instance().start()