import psutil
import time
from typing import Dict, List
import json
from pathlib import Path

class DataHoarderDetector:
    def __init__(self, config: Dict):
        self.config = config
        self.usage_history = []
        self.warning_threshold = 0.7  # 70% of expected contribution
        self.penalty_threshold = 0.5  # 50% of expected contribution
        
    def analyze_user_contribution(self, user_id: str) -> Dict:
        """Analyze user's resource contribution patterns"""
        current_usage = self._get_current_usage()
        expected = self._get_expected_contribution()
        contribution_ratio = self._calculate_contribution_ratio(current_usage, expected)
        
        self.usage_history.append({
            'user_id': user_id,
            'timestamp': time.time(),
            'usage': current_usage,
            'ratio': contribution_ratio
        })
        
        return self._generate_contribution_report(user_id, contribution_ratio)
    
    def _get_current_usage(self) -> Dict:
        """Get current system resource usage"""
        return {
            'cpu_cores': psutil.cpu_count(),
            'cpu_percent': psutil.cpu_percent(),
            'ram_gb': psutil.virtual_memory().total / (1024**3),
            'ram_percent': psutil.virtual_memory().percent,
            'storage_gb': psutil.disk_usage('/').total / (1024**3),
            'storage_percent': psutil.disk_usage('/').percent
        }
    
    def _get_expected_contribution(self) -> Dict:
        """Get expected resource contribution based on system config"""
        return self.config['hpc']['min_contribution']
    
    def _calculate_contribution_ratio(self, current: Dict, expected: Dict) -> float:
        """Calculate the ratio of actual to expected contribution"""
        ratios = []
        if current['cpu_cores'] > 0:
            ratios.append(current['cpu_cores'] / expected['cpu_cores'])
        if current['ram_gb'] > 0:
            ratios.append(current['ram_gb'] / expected['ram_gb'])
        if current['storage_gb'] > 0:
            ratios.append(current['storage_gb'] / expected['storage_gb'])
            
        return sum(ratios) / len(ratios) if ratios else 0.0
    
    def _generate_contribution_report(self, user_id: str, ratio: float) -> Dict:
        """Generate a report on user's resource contribution"""
        status = 'good'
        action = None
        fee = 0.0
        
        if ratio < self.penalty_threshold:
            status = 'penalty'
            action = 'apply_fee'
            fee = self._calculate_penalty_fee(ratio)
        elif ratio < self.warning_threshold:
            status = 'warning'
            action = 'send_warning'
            
        return {
            'user_id': user_id,
            'status': status,
            'contribution_ratio': ratio,
            'action_required': action,
            'penalty_fee': fee,
            'timestamp': time.time()
        }
    
    def _calculate_penalty_fee(self, ratio: float) -> float:
        """Calculate penalty fee based on contribution ratio"""
        base_fee = 10.0  # Base fee in credits/tokens
        shortfall = self.penalty_threshold - ratio
        return base_fee * (shortfall / self.penalty_threshold)
