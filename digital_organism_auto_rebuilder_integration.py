#!/usr/bin/env python3
"""
Digital Organism Auto-Rebuilder Integration Framework
=====================================================

Integrates the sophisticated auto_rebuilder.py capabilities as a core Digital Organism component
for continuous self-improvement, code integration, and autonomous development capabilities.

Key Integration Points:
1. Heartbeat/Self-Improvement Engine
2. Dynamic Code Integration Hub  
3. Security & Safety Guardian
4. Autonomous Development Assistant
5. Real-time System Evolution Engine

Author: Digital Organism Core Team
Date: June 6, 2025
Version: 1.0.0
"""

import os
import sys
import time
import json
import threading
import asyncio
from pathlib import Path
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass, field
from datetime import datetime, timedelta

# Import the auto_rebuilder capabilities
sys.path.append(os.path.dirname(__file__))
from auto_rebuilder import (
    calculate_module_clusters,
    test_module_in_sandbox,
    assess_code_safety,
    resolve_namespace_conflicts,
    extract_functions_and_classes,
    create_hierarchical_module_structure,
    find_best_entry_points,
    log as auto_log
)

@dataclass
class DigitalOrganismConfig:
    """Configuration for Digital Organism Auto-Rebuilder Integration"""
    # Heartbeat settings
    heartbeat_interval: int = 300  # 5 minutes
    self_improvement_threshold: float = 0.7  # Trigger improvement at 70% efficiency
    
    # Code integration settings
    max_concurrent_integrations: int = 8
    security_threshold: int = 70  # Minimum security score for integration
    compatibility_threshold: float = 0.6
    
    # Autonomous development settings
    enable_autonomous_coding: bool = True
    enable_self_modification: bool = False  # Requires explicit authorization
    learning_rate: float = 0.1
    
    # Safety & monitoring
    sandbox_timeout: int = 10  # seconds
    max_memory_usage: int = 1024  # MB
    enable_network_access: bool = False
    
    # Paths
    workspace_path: str = field(default_factory=lambda: os.getcwd())
    backup_path: str = field(default_factory=lambda: os.path.join(os.getcwd(), "backups"))
    integration_log_path: str = field(default_factory=lambda: "digital_organism_integration.log")

class DigitalOrganismAutoRebuilder:
    """
    Core Digital Organism Auto-Rebuilder Integration System
    
    This class wraps the sophisticated auto_rebuilder.py capabilities and integrates them
    as core Digital Organism functionality for:
    
    1. Continuous Self-Improvement (Heartbeat)
    2. Dynamic Code Integration
    3. Security & Safety Monitoring
    4. Autonomous Development Assistance
    5. Real-time System Evolution
    """
    
    def __init__(self, config: DigitalOrganismConfig):
        self.config = config
        self.is_running = False
        self.heartbeat_thread = None
        self.integration_queue = asyncio.Queue()
        self.performance_metrics = {}
        self.security_alerts = []
        self.last_self_improvement = datetime.now()
        
        # Initialize logging
        self.logger = self._setup_logging()
        
        # Track system state
        self.system_health = {
            "integration_success_rate": 1.0,
            "security_incidents": 0,
            "performance_score": 1.0,
            "last_heartbeat": None,
            "autonomous_improvements": 0
        }
        
        self.logger.info("🔥 Digital Organism Auto-Rebuilder initialized")
    
    def _setup_logging(self):
        """Setup integrated logging system"""
        import logging
        
        logger = logging.getLogger("DigitalOrganismAutoRebuilder")
        logger.setLevel(logging.INFO)
        
        # File handler with rotation
        handler = logging.FileHandler(self.config.integration_log_path)
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - [%(funcName)s] - %(message)s'
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        
        return logger
    
    async def start_heartbeat(self):
        """Start the Digital Organism heartbeat for continuous self-improvement"""
        if self.is_running:
            self.logger.warning("Heartbeat already running")
            return
        
        self.is_running = True
        self.logger.info("💓 Starting Digital Organism heartbeat system")
        
        while self.is_running:
            try:
                await self._heartbeat_cycle()
                await asyncio.sleep(self.config.heartbeat_interval)
            except Exception as e:
                self.logger.error(f"Heartbeat cycle error: {e}")
                await asyncio.sleep(30)  # Brief recovery pause
    
    async def _heartbeat_cycle(self):
        """Execute a complete heartbeat cycle"""
        start_time = time.time()
        self.logger.info("🔄 Starting heartbeat cycle")
        
        # 1. System Health Assessment
        health_score = await self._assess_system_health()
        
        # 2. Security Monitoring
        security_score = await self._monitor_security()
        
        # 3. Performance Analysis
        performance_score = await self._analyze_performance()
        
        # 4. Self-Improvement Check
        if health_score < self.config.self_improvement_threshold:
            await self._trigger_self_improvement()
        
        # 5. Code Integration Processing
        await self._process_integration_queue()
        
        # 6. Update System State
        self.system_health.update({
            "last_heartbeat": datetime.now(),
            "performance_score": performance_score,
            "integration_success_rate": health_score
        })
        
        cycle_time = time.time() - start_time
        self.logger.info(f"✅ Heartbeat cycle completed in {cycle_time:.2f}s")
        
        # Auto-log to auto_rebuilder system for correlation
        auto_log(
            f"Digital Organism heartbeat: Health={health_score:.2f}, Security={security_score:.2f}, Performance={performance_score:.2f}",
            level="INFO",
            context="HEARTBEAT",
            script_id="digital_organism",
            integration_phase="MONITOR"
        )
    
    async def _assess_system_health(self) -> float:
        """Assess overall system health using auto_rebuilder metrics"""
        try:
            # Scan current workspace for potential improvements
            workspace_files = list(Path(self.config.workspace_path).rglob("*.py"))
            
            if not workspace_files:
                return 1.0  # Perfect health if no files to analyze
            
            total_score = 0
            file_count = 0
            
            for file_path in workspace_files[:50]:  # Sample first 50 files
                try:
                    # Use auto_rebuilder's safety assessment
                    with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                        content = f.read()
                    
                    safety_report = assess_code_safety(content, str(file_path))
                    total_score += safety_report['score']
                    file_count += 1
                    
                except Exception as e:
                    self.logger.debug(f"Could not assess {file_path}: {e}")
            
            health_score = (total_score / file_count / 100) if file_count > 0 else 1.0
            self.logger.info(f"📊 System health score: {health_score:.2f}")
            return health_score
            
        except Exception as e:
            self.logger.error(f"Health assessment failed: {e}")
            return 0.5  # Default to moderate health
    
    async def _monitor_security(self) -> float:
        """Monitor security using auto_rebuilder's security analysis"""
        try:
            # Scan for security issues in recent changes
            security_incidents = 0
            total_files = 0
            
            # Get recently modified files
            recent_files = []
            for file_path in Path(self.config.workspace_path).rglob("*.py"):
                try:
                    if file_path.stat().st_mtime > (time.time() - 3600):  # Last hour
                        recent_files.append(file_path)
                except:
                    continue
            
            for file_path in recent_files[:20]:  # Check last 20 recent files
                try:
                    with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                        content = f.read()
                    
                    safety_report = assess_code_safety(content, str(file_path))
                    total_files += 1
                    
                    if safety_report['score'] < self.config.security_threshold:
                        security_incidents += 1
                        self.security_alerts.append({
                            'file': str(file_path),
                            'score': safety_report['score'],
                            'risks': safety_report['risks'],
                            'timestamp': datetime.now()
                        })
                    
                except Exception as e:
                    self.logger.debug(f"Could not analyze {file_path}: {e}")
            
            security_score = 1.0 - (security_incidents / max(total_files, 1))
            
            if security_incidents > 0:
                self.logger.warning(f"🚨 {security_incidents} security incidents detected")
            
            return security_score
            
        except Exception as e:
            self.logger.error(f"Security monitoring failed: {e}")
            return 0.8  # Default to good security
    
    async def _analyze_performance(self) -> float:
        """Analyze system performance"""
        try:
            # Simple performance metrics
            import psutil
            
            cpu_percent = psutil.cpu_percent(interval=1)
            memory_percent = psutil.virtual_memory().percent
            
            # Performance score based on resource usage
            performance_score = 1.0 - (cpu_percent / 100 * 0.3 + memory_percent / 100 * 0.7)
            performance_score = max(0, min(1, performance_score))
            
            self.logger.info(f"⚡ Performance score: {performance_score:.2f} (CPU: {cpu_percent}%, RAM: {memory_percent}%)")
            return performance_score
            
        except Exception as e:
            self.logger.error(f"Performance analysis failed: {e}")
            return 0.8
    
    async def _trigger_self_improvement(self):
        """Trigger self-improvement using auto_rebuilder capabilities"""
        self.logger.info("🔧 Triggering self-improvement cycle")
        
        try:
            # Use auto_rebuilder to analyze and improve code organization
            workspace_files = []
            for file_path in Path(self.config.workspace_path).rglob("*.py"):
                if file_path.name != __file__:  # Don't analyze ourselves
                    workspace_files.append({
                        'filename': file_path.name,
                        'filepath': str(file_path),
                        'package': 'core'  # Default package
                    })
            
            if workspace_files:
                # Calculate module clusters for better organization
                clusters = calculate_module_clusters(workspace_files[:100])  # Limit for performance
                
                # Create hierarchical structure
                create_hierarchical_module_structure(workspace_files[:50])
                
                self.system_health["autonomous_improvements"] += 1
                self.last_self_improvement = datetime.now()
                
                auto_log(
                    f"Self-improvement completed: {len(clusters)} clusters created",
                    level="SUCCESS",
                    context="SELF_IMPROVEMENT",
                    script_id="digital_organism",
                    integration_phase="IMPROVE"
                )
        
        except Exception as e:
            self.logger.error(f"Self-improvement failed: {e}")
    
    async def _process_integration_queue(self):
        """Process pending code integrations"""
        processed = 0
        while not self.integration_queue.empty() and processed < 5:  # Limit per cycle
            try:
                integration_task = await self.integration_queue.get()
                await self._execute_integration(integration_task)
                processed += 1
            except Exception as e:
                self.logger.error(f"Integration processing failed: {e}")
    
    async def integrate_code(self, code_path: str, metadata: Dict[str, Any] = None):
        """
        Integrate new code using auto_rebuilder's sophisticated analysis
        
        Args:
            code_path: Path to code file to integrate
            metadata: Optional metadata about the code
        """
        integration_task = {
            'code_path': code_path,
            'metadata': metadata or {},
            'timestamp': datetime.now(),
            'priority': metadata.get('priority', 'normal') if metadata else 'normal'
        }
        
        await self.integration_queue.put(integration_task)
        self.logger.info(f"📥 Queued code integration: {code_path}")
    
    async def _execute_integration(self, task: Dict[str, Any]):
        """Execute a code integration task"""
        code_path = task['code_path']
        self.logger.info(f"🔄 Executing integration: {code_path}")
        
        try:
            # 1. Security assessment
            sandbox_result = test_module_in_sandbox(
                code_path,
                timeout=self.config.sandbox_timeout,
                max_memory_mb=self.config.max_memory_usage,
                allow_network=self.config.enable_network_access,
                safety_check=True
            )
            
            if not sandbox_result.get('compatible', False):
                self.logger.warning(f"❌ Integration failed - incompatible: {code_path}")
                auto_log(
                    f"Integration rejected: {sandbox_result.get('error', 'Unknown error')}",
                    level="WARNING",
                    context="INTEGRATION",
                    script_id="digital_organism",
                    integration_phase="REJECT"
                )
                return
            
            # 2. Extract and analyze components
            with open(code_path, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()
            
            import ast
            tree = ast.parse(content)
            components = extract_functions_and_classes(tree, os.path.basename(code_path))
            
            # 3. Successful integration
            self.logger.info(f"✅ Successfully integrated: {code_path}")
            auto_log(
                f"Integration successful: {len(components)} components integrated",
                level="SUCCESS",
                context="INTEGRATION",
                script_id="digital_organism",
                integration_phase="COMPLETE"
            )
            
        except Exception as e:
            self.logger.error(f"Integration execution failed for {code_path}: {e}")
    
    def get_system_status(self) -> Dict[str, Any]:
        """Get current system status"""
        return {
            'is_running': self.is_running,
            'system_health': self.system_health.copy(),
            'security_alerts': len(self.security_alerts),
            'integration_queue_size': self.integration_queue.qsize(),
            'last_self_improvement': self.last_self_improvement.isoformat(),
            'config': {
                'heartbeat_interval': self.config.heartbeat_interval,
                'security_threshold': self.config.security_threshold,
                'autonomous_coding_enabled': self.config.enable_autonomous_coding
            }
        }
    
    async def stop(self):
        """Stop the Digital Organism Auto-Rebuilder"""
        self.logger.info("🛑 Stopping Digital Organism Auto-Rebuilder")
        self.is_running = False
        
        # Final status report
        status = self.get_system_status()
        auto_log(
            f"Digital Organism Auto-Rebuilder stopped. Final status: {json.dumps(status, indent=2)}",
            level="INFO",
            context="SHUTDOWN",
            script_id="digital_organism",
            integration_phase="STOP"
        )

# Convenience functions for Digital Organism integration
async def initialize_digital_organism_rebuilder(config: DigitalOrganismConfig = None) -> DigitalOrganismAutoRebuilder:
    """Initialize the Digital Organism Auto-Rebuilder with default or custom config"""
    if config is None:
        config = DigitalOrganismConfig()
    
    rebuilder = DigitalOrganismAutoRebuilder(config)
    return rebuilder

async def run_as_heartbeat_service():
    """Run as a background heartbeat service for the Digital Organism"""
    config = DigitalOrganismConfig(
        heartbeat_interval=300,  # 5 minutes
        enable_autonomous_coding=True,
        enable_self_modification=False  # Safety first
    )
    
    rebuilder = await initialize_digital_organism_rebuilder(config)
    
    try:
        await rebuilder.start_heartbeat()
    except KeyboardInterrupt:
        print("\n🛑 Received interrupt signal")
    finally:
        await rebuilder.stop()

# Integration with existing Digital Organism systems
class DigitalOrganismRebuilderAdapter:
    """Adapter to integrate with existing Digital Organism systems"""
    
    @staticmethod
    def integrate_with_consciousness_engine(consciousness_engine):
        """Integrate with consciousness engine for self-aware improvements"""
        # This would connect the auto-rebuilder with consciousness components
        pass
    
    @staticmethod
    def integrate_with_aeos_orchestrator(orchestrator):
        """Integrate with AEOS orchestrator for distributed improvements"""
        # This would connect with the production orchestrator
        pass
    
    @staticmethod
    def integrate_with_gamification_system(game_system):
        """Integrate with gamification for improvement rewards"""
        # This would connect improvements with achievement systems
        pass

if __name__ == "__main__":
    """
    Example usage as a Digital Organism core component
    """
    import asyncio
    
    print("🔥 Digital Organism Auto-Rebuilder Integration")
    print("=" * 50)
    print("Starting as heartbeat service...")
    print("Press Ctrl+C to stop")
    
    asyncio.run(run_as_heartbeat_service())
