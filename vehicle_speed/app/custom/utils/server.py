import http.server
import socketserver
import threading
import time

import requests
from functools import partial

class TimeoutTCPServer(socketserver.TCPServer):
    def __init__(self, server_address, RequestHandlerClass, timeout=30):
        super().__init__(server_address, RequestHandlerClass)
        self.timeout = timeout
        self.last_request_time = time.time()
        self._shutdown_monitor = threading.Thread(target=self._monitor_timeout, daemon=True)
        self._shutdown_monitor.start()

    def _monitor_timeout(self):
        while True:
            time.sleep(1)
            elapsed = time.time() - self.last_request_time
            if self.timeout is not None and elapsed > self.timeout:
                print(f" Timeout di {self.timeout}s raggiunto, arresto del server.")
                self.shutdown()
                break

class CustomHandler(http.server.SimpleHTTPRequestHandler):
    server: TimeoutTCPServer  # type: ignore

    def do_GET(self):
        # aggiorna il timer di attività
        self.server.last_request_time = time.time()
        return super().do_GET()

def start_data_server(data_dir, port=8000, timeout=60):
    handler = partial(CustomHandler, directory=data_dir)
    httpd = TimeoutTCPServer(("", port), handler, timeout=timeout)

    server_thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    server_thread.start()

    print(f" Server avviato su porta {port}, timeout inattività: {timeout}s")
    return httpd


def import_data_from_url(url):
    response = requests.get(url)
    from io import StringIO
    return StringIO(response.text)