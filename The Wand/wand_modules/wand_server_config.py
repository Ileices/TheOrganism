from waitress import serve
import logging
from typing import Any
from flask import Flask

class ProductionServer:
    def __init__(self, app: Flask, host: str = "0.0.0.0", port: int = 6000):
        self.app = app
        self.host = host
        self.port = port
        self.logger = logging.getLogger(__name__)
        
    def run(self):
        """Run Flask app using production server"""
        try:
            self.logger.info(f"Starting production server on {self.host}:{self.port}")
            serve(self.app, host=self.host, port=self.port)
        except Exception as e:
            self.logger.error(f"Server failed to start: {e}")
            raise
