import http.server
import socketserver
import threading
import os

import requests

def start_data_server(data_dir, port=8000):
    os.chdir(data_dir)
    handler = http.server.SimpleHTTPRequestHandler
    httpd = socketserver.TCPServer(("", port), handler)
    
    # Server in background
    server_thread = threading.Thread(target=httpd.serve_forever)
    server_thread.daemon = True
    server_thread.start()
    
    print("Server started")
    return httpd

def load_data_from_url(url):
    response = requests.get(url)
    from io import StringIO
    return StringIO(response.text)