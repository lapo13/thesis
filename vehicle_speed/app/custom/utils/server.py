import http.server
import socketserver
import threading
import os

import requests
from functools import partial

def start_data_server(data_dir, port=8000):
    handler = partial(http.server.SimpleHTTPRequestHandler, directory=data_dir)
    httpd = socketserver.TCPServer(("", port), handler)

    server_thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    server_thread.start()

    print(f"Server started on port {port}, serving: {data_dir}")
    return httpd


def load_data_from_url(url):
    response = requests.get(url)
    from io import StringIO
    return StringIO(response.text)