import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
import logging
from typing import Optional, Dict

class DatasetService:
    def __init__(self, base_url: str = "http://127.0.0.1:5002"):
        self.base_url = base_url
        self.logger = logging.getLogger(__name__)
        self.session = self._create_retry_session()
        
    def _create_retry_session(self):
        """Create requests session with retry logic"""
        session = requests.Session()
        retries = Retry(
            total=5,
            backoff_factor=0.5,
            status_forcelist=[500, 502, 503, 504]
        )
        session.mount('http://', HTTPAdapter(max_retries=retries))
        return session
        
    def download_dataset(self) -> Optional[Dict]:
        """Download dataset with retry mechanism"""
        try:
            response = self.session.get(f"{self.base_url}/download_dataset", timeout=30)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            self.logger.error(f"Dataset download failed: {e}")
            return None
