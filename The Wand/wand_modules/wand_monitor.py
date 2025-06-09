import psutil
import threading
import time
from typing import Dict, List
from queue import Queue
import logging
from copy import deepcopy

class ResourceMonitor:
    """Monitor system resources and provide real-time metrics"""
    def __init__(self):
        self.metrics = {
            'cpu': {},
            'memory': {},
            'disk': {},
            'network': {}
        }
        self._monitoring = False
        self._update_interval = 1.0  # seconds
        
    def start(self):
        """Start resource monitoring"""
        self._monitoring = True
        threading.Thread(target=self._monitor_loop, daemon=True).start()
        
    def stop(self):
        """Stop resource monitoring"""
        self._monitoring = False
        
    def get_metrics(self) -> Dict:
        """Get current resource metrics"""
        return self.metrics.copy()
        
    def _monitor_loop(self):
        """Main monitoring loop"""
        while self._monitoring:
            try:
                # Update CPU metrics
                self.metrics['cpu'] = {
                    'percent': psutil.cpu_percent(interval=None),
                    'count': psutil.cpu_count(),
                    'frequency': psutil.cpu_freq()._asdict() if psutil.cpu_freq() else {},
                }
                
                # Update memory metrics
                mem = psutil.virtual_memory()
                self.metrics['memory'] = {
                    'total': mem.total,
                    'available': mem.available,
                    'percent': mem.percent,
                    'used': mem.used,
                }
                
                # Update disk metrics
                disk = psutil.disk_usage('/')
                self.metrics['disk'] = {
                    'total': disk.total,
                    'used': disk.used,
                    'free': disk.free,
                    'percent': disk.percent,
                }
                
                # Update network metrics
                net = psutil.net_io_counters()
                self.metrics['network'] = {
                    'bytes_sent': net.bytes_sent,
                    'bytes_recv': net.bytes_recv,
                    'packets_sent': net.packets_sent,
                    'packets_recv': net.packets_recv,
                }
                
            except Exception as e:
                logging.error(f"Resource monitoring error: {e}")
                
            time.sleep(self._update_interval)

class WandMonitor:
    def __init__(self):
        self.metrics = {}
        self._monitoring = False
        self.components = {}
        self.component_health = {}
        self.last_health_check = {}
        self.health_check_interval = 5  # seconds
        self.failure_thresholds = {
            'consecutive_failures': 3,
            'timeframe_failures': 5,
            'timeframe_minutes': 30
        }
        self.failure_counts = {}
        self.retry_configurations = {}
        self.max_global_retries = 3
        self.active_monitoring_rules = {}
        self.recovery_queue = Queue()
        self.execution_metrics = {}
        self.performance_thresholds = {}
        self.resource_monitor = ResourceMonitor()
        self.active_builds = {}
        self.execution_states = {}
        self.paused_builds = set()
        self.prediction_log = {}
        self.intervention_history = {}
        self.system_performance = {}
        self.resource_history = {}
        self.system_metrics = {}
        self.workload_distribution = {}
        self.execution_drift = {}
        
    def register_component(self, name: str, component: object):
        """Register a component for monitoring with health checks"""
        self.components[name] = component
        self.metrics[name] = {
            'status': 'registered',
            'last_update': time.time(),
            'errors': [],
            'health': 'unknown'
        }
        self.component_health[name] = True
        self.last_health_check[name] = time.time()
    
    def start_monitoring(self):
        """Start system monitoring"""
        self._monitoring = True
        threading.Thread(target=self._monitor_loop, daemon=True).start()
        
    def _monitor_loop(self):
        """Enhanced monitoring loop with health checks"""
        while self._monitoring:
            current_time = time.time()
            
            # Monitor system resources
            self.metrics['system'] = {
                'cpu_percent': psutil.cpu_percent(),
                'memory_percent': psutil.virtual_memory().percent,
                'timestamp': current_time
            }
            
            # Monitor component health
            for name, component in self.components.items():
                try:
                    if current_time - self.last_health_check[name] >= self.health_check_interval:
                        self._check_component_health(name, component)
                        self.last_health_check[name] = current_time
                        
                    # Update component metrics
                    if hasattr(component, 'get_status'):
                        self.metrics[name]['status'] = component.get_status()
                    self.metrics[name]['last_update'] = current_time
                    
                except Exception as e:
                    self._handle_component_error(name, e)
            
            time.sleep(1)
            
    def _check_component_health(self, name: str, component: object):
        """Check if component is responsive and functioning"""
        try:
            # Basic health check - verify component is responsive
            if hasattr(component, 'health_check'):
                health_status = component.health_check()
            else:
                health_status = bool(component)
                
            # Update health status
            self.component_health[name] = health_status
            self.metrics[name]['health'] = 'healthy' if health_status else 'unhealthy'
            
        except Exception as e:
            self._handle_component_error(name, e)
            
    def _handle_component_error(self, name: str, error: Exception):
        """Enhanced error handling with AI-driven recovery"""
        self.component_health[name] = False
        self.metrics[name].update({
            'status': 'error',
            'health': 'unhealthy',
            'last_error': str(error),
            'last_error_time': time.time()
        })
        self.metrics[name]['errors'].append({
            'time': time.time(),
            'error': str(error)
        })
        logging.error(f"Component '{name}' error: {error}")
        
        # Track failure frequency
        current_time = time.time()
        if name not in self.failure_counts:
            self.failure_counts[name] = []
            
        self.failure_counts[name].append(current_time)
        
        # Clean up old failures
        cutoff_time = current_time - (self.failure_thresholds['timeframe_minutes'] * 60)
        self.failure_counts[name] = [t for t in self.failure_counts[name] if t > cutoff_time]
        
        # Check failure thresholds
        if self._check_failure_thresholds(name):
            self._trigger_critical_failure_alert(name)
            
        # If this is a build failure and we have AI suggestions
        if name == 'builder' and hasattr(self, 'ai'):
            try:
                failed_build = self.components[name].current_build
                if failed_build:
                    retry_strategy = self.ai.suggest_retry_strategy(failed_build)
                    if retry_strategy:
                        self._attempt_smart_retry(name, failed_build, retry_strategy)
                        return
            except Exception as e:
                logging.error(f"Smart retry analysis failed: {e}")
                
        # Check for AI recovery strategies before standard handling
        if hasattr(self, 'ai'):
            recovery_strategy = self._get_ai_recovery_strategy(name, error)
            if recovery_strategy:
                success = self._execute_recovery_strategy(name, recovery_strategy)
                if success:
                    return
                    
        # Request AI recovery strategy
        if hasattr(self, 'ai'):
            recovery_plan = self.ai.generate_recovery_plan(name, error, self.metrics[name])
            if recovery_plan and self._attempt_recovery(name, recovery_plan):
                return
                
        self._trigger_fallback_recovery(name, error)
        
    def _attempt_smart_retry(self, component_name: str, failed_build: Dict, strategy: Dict):
        """Attempt retry with AI-suggested modifications"""
        if not strategy['modifications']:
            return
            
        max_retries = min(strategy['max_retries'], self.max_global_retries)
        current_retries = len(self.failure_counts.get(component_name, []))
        
        if current_retries >= max_retries:
            self._trigger_critical_failure_alert(component_name)
            return
            
        # Apply modifications and retry
        modified_build = self._apply_retry_modifications(failed_build, strategy['modifications'])
        
        # Log retry attempt
        self.logger.info(
            f"Attempting smart retry {current_retries + 1}/{max_retries} "
            f"for {component_name} with confidence {strategy['confidence']:.2f}"
        )
        
        # Queue modified build
        if hasattr(self, 'core'):
            self.core.queue_build(modified_build, priority=True)
            
    def _apply_retry_modifications(self, build_data: Dict, modifications: List[Dict]) -> Dict:
        """Apply AI-suggested modifications to build data"""
        modified_build = deepcopy(build_data)
        
        for mod in modifications:
            try:
                if mod['type'] == 'task_order':
                    modified_build['tasks'] = self._reorder_tasks(
                        modified_build['tasks'], 
                        mod['suggested_order']
                    )
                elif mod['type'] == 'parameter_adjust':
                    self._adjust_build_parameters(
                        modified_build, 
                        mod['parameters']
                    )
            except Exception as e:
                logging.error(f"Failed to apply modification: {e}")
                
        return modified_build
        
    def _execute_recovery_strategy(self, component_name: str, strategy: Dict) -> bool:
        """Execute AI-suggested recovery strategy"""
        try:
            # Apply suggested modifications
            if 'parameter_adjustments' in strategy:
                self._apply_parameter_adjustments(component_name, strategy['parameter_adjustments'])
                
            # Reorder task execution if suggested
            if 'task_reordering' in strategy:
                self._reorder_component_tasks(component_name, strategy['task_reordering'])
                
            # Apply resource adjustments
            if 'resource_adjustments' in strategy:
                self._adjust_component_resources(component_name, strategy['resource_adjustments'])
                
            # Monitor recovery execution
            recovery_success = self._monitor_recovery_execution(component_name, strategy)
            
            if recovery_success:
                self.logger.info(f"AI recovery successful for {component_name}")
                return True
                
        except Exception as e:
            self.logger.error(f"AI recovery failed for {component_name}: {e}")
            
        return False
        
    def _monitor_recovery_execution(self, component_name: str, strategy: Dict) -> bool:
        """Monitor the execution of a recovery strategy"""
        monitoring_rules = strategy.get('monitoring_rules', [])
        violation_count = 0
        
        # Set up monitoring timeframe
        end_time = time.time() + strategy.get('monitoring_duration', 300)  # 5 minutes default
        
        while time.time() < end_time:
            # Check each monitoring rule
            for rule in monitoring_rules:
                if not self._check_monitoring_rule(component_name, rule):
                    violation_count += 1
                    if violation_count >= rule.get('max_violations', 3):
                        return False
                        
            time.sleep(rule.get('check_interval', 1))
            
        return True
        
    def _check_failure_thresholds(self, component_name: str) -> bool:
        """Check if component has exceeded failure thresholds"""
        failures = self.failure_counts.get(component_name, [])
        
        # Check consecutive failures
        if len(failures) >= self.failure_thresholds['consecutive_failures']:
            return True
            
        # Check failures within timeframe
        current_time = time.time()
        timeframe = self.failure_thresholds['timeframe_minutes'] * 60
        recent_failures = [f for f in failures if current_time - f <= timeframe]
        
        return len(recent_failures) >= self.failure_thresholds['timeframe_failures']
        
    def _trigger_critical_failure_alert(self, component_name: str):
        """Handle critical failure detection"""
        alert = {
            'component': component_name,
            'failure_count': len(self.failure_counts[component_name]),
            'timestamp': time.time(),
            'status': self.metrics[component_name]
        }
        
        # Log critical failure
        logging.critical(f"Critical failure detected in {component_name}")
        
        # Notify core system
        if hasattr(self, 'core'):
            self.core.handle_critical_failure(alert)
            
    def _attempt_recovery(self, component_name: str, recovery_plan: Dict) -> bool:
        """Execute AI-suggested recovery plan"""
        try:
            # Apply suggested modifications
            if recovery_plan.get('parameter_adjustments'):
                self._apply_parameter_adjustments(component_name, 
                                               recovery_plan['parameter_adjustments'])
                
            # Modify execution strategy if suggested
            if recovery_plan.get('execution_strategy'):
                self._modify_execution_strategy(component_name, 
                                             recovery_plan['execution_strategy'])
                
            # Set up monitoring rules
            monitoring_rules = recovery_plan.get('monitoring_rules', [])
            return self._monitor_recovery(component_name, monitoring_rules)
            
        except Exception as e:
            logging.error(f"Recovery attempt failed for {component_name}: {e}")
            return False
            
    def _monitor_recovery(self, component_name: str, rules: List[Dict]) -> bool:
        """Monitor component recovery with AI-provided rules"""
        start_time = time.time()
        max_duration = 300  # 5 minutes max for recovery
        
        while time.time() - start_time < max_duration:
            violations = []
            
            for rule in rules:
                if not self._check_rule_compliance(component_name, rule):
                    violations.append(rule['id'])
                    
            if not violations:
                return True  # Recovery successful
                
            if len(violations) > 2:  # Too many violations
                break
                
            time.sleep(1)
            
        return False
        
    def _monitor_build_execution(self, build_id: str, execution_plan: Dict):
        """Monitor build execution with real-time adjustments"""
        self.execution_metrics[build_id] = {
            'start_time': time.time(),
            'resource_usage': [],
            'task_completion': {},
            'performance_issues': []
        }
        
        while self._is_build_running(build_id):
            metrics = self._collect_execution_metrics(build_id)
            
            # Check for performance issues
            if self._detect_performance_issues(metrics):
                self._apply_runtime_adjustments(build_id)
                
            # Update execution metrics
            self._update_execution_metrics(build_id, metrics)
            
            time.sleep(1)  # Monitoring interval
            
    def _apply_runtime_adjustments(self, build_id: str):
        """Apply AI-suggested runtime adjustments"""
        metrics = self.execution_metrics[build_id]
        
        if hasattr(self, 'ai'):
            adjustments = self.ai.suggest_runtime_adjustments(metrics)
            
            if adjustments:
                self._apply_resource_adjustments(build_id, adjustments.get('resource_adjustment'))
                self._modify_execution_plan(build_id, adjustments.get('execution_modification'))
                
    def _detect_performance_issues(self, metrics: Dict) -> bool:
        """Detect performance issues in real-time"""
        issues = []
        
        # Check resource utilization
        if metrics['cpu_usage'] > 90:
            issues.append('high_cpu')
        if metrics['memory_usage'] > 85:
            issues.append('high_memory')
            
        # Check task progression
        if self._is_task_progression_slow(metrics):
            issues.append('slow_progression')
            
        return bool(issues)
        
    def _optimize_running_build(self, build_id: str):
        """Optimize currently running build based on metrics"""
        current_metrics = self.execution_metrics[build_id]
        
        # Get AI optimization suggestions
        if hasattr(self, 'ai'):
            optimizations = self.ai.optimize_execution_plan({
                'metrics': current_metrics,
                'build_id': build_id
            })
            
            if optimizations:
                self._apply_execution_optimizations(build_id, optimizations)
                
    def _monitor_execution(self, build_id: str):
        """Enhanced execution monitoring with real-time intervention"""
        self.execution_states[build_id] = self._initialize_execution_state(build_id)
        
        while self._is_build_active(build_id):
            metrics = self._collect_execution_metrics(build_id)
            
            # Check for potential failures
            if hasattr(self, 'ai'):
                prediction = self.ai.predict_execution_failures(build_id, metrics)
                if prediction['potential_failures']:
                    self._handle_predicted_failure(build_id, prediction)
                    
            # Check performance and resources
            performance = self._check_execution_performance(build_id, metrics)
            if not performance['acceptable']:
                self._handle_performance_issues(build_id, performance)
                
            time.sleep(self.monitoring_interval)
            
    def _handle_predicted_failure(self, build_id: str, prediction: Dict):
        """Enhanced failure prevention with AI authority"""
        if prediction['confidence'] > 0.8:  # High confidence threshold
            # Pause execution
            self._pause_execution(build_id)
            self.paused_builds.add(build_id)
            
            try:
                # Get AI intervention strategy
                intervention = self.ai.generate_intervention_strategy(build_id, prediction)
                
                if intervention['confidence'] > 0.85:
                    # Allow AI to modify build parameters
                    self._apply_parameter_overrides(build_id, intervention['parameter_adjustments'])
                    
                    # Inject new execution logic if needed
                    if intervention['execution_modifications']:
                        self._inject_execution_logic(build_id, intervention['execution_modifications'])
                        
                    # Resume with modified execution
                    success = self._resume_with_modifications(build_id, intervention)
                    
                    # Log intervention results
                    self._log_intervention_result(build_id, intervention, success)
                    
                    if success:
                        self.paused_builds.remove(build_id)
                        return
                        
            except Exception as e:
                logging.error(f"AI intervention failed: {e}")
                
            self._trigger_emergency_shutdown(build_id)
            
    def _inject_execution_logic(self, build_id: str, modifications: Dict):
        """Inject new execution logic during runtime"""
        try:
            # Validate modifications
            if self._validate_logic_modifications(modifications):
                # Create backup of current state
                self._backup_execution_state(build_id)
                
                # Apply new logic
                self._apply_logic_modifications(build_id, modifications)
                
                self.logger.info(f"Injected new execution logic for build {build_id}")
            else:
                raise ValueError("Invalid logic modifications")
                
        except Exception as e:
            self.logger.error(f"Failed to inject execution logic: {e}")
            self._restore_execution_state(build_id)
            
    def _handle_performance_issues(self, build_id: str, performance: Dict):
        """Handle performance issues with dynamic optimization"""
        if not hasattr(self, 'ai'):
            return
            
        try:
            # Get optimization suggestions
            optimizations = self.ai.suggest_execution_improvements(performance)
            if not optimizations:
                return
                
            # Apply optimizations in real-time
            self._apply_runtime_optimizations(build_id, optimizations)
            
        except Exception as e:
            logging.error(f"Performance optimization failed: {e}")
            
    def _monitor_execution(self, build_id: str):
        """Monitor execution with AI-driven intervention"""
        metrics = self._initialize_execution_metrics(build_id)
        
        while self._is_build_active(build_id):
            current_metrics = self._collect_live_metrics(build_id)
            
            # Check for potential failures
            if self._should_check_prediction(metrics):
                prediction = self.ai.predict_execution_outcome(current_metrics)
                if prediction['success_probability'] < 0.7:  # Risk threshold
                    self._handle_predicted_failure(build_id, prediction)
                    
            # Monitor system resources
            if self._detect_resource_constraint(current_metrics):
                self._rebalance_execution(build_id, current_metrics)
                
            # Update execution metrics
            metrics = self._update_metrics(metrics, current_metrics)
            time.sleep(self.monitor_interval)
            
    def _rebalance_execution(self, build_id: str, metrics: Dict):
        """Dynamically rebalance execution based on system resources"""
        try:
            # Pause execution
            self._pause_build_execution(build_id)
            
            # Get AI-suggested adaptations
            adaptations = self.ai.adapt_to_system_load(build_id, metrics)
            
            if adaptations['confidence'] > 0.8:
                # Apply resource adjustments
                self._apply_resource_adjustments(build_id, adaptations['resource_adjustments'])
                
                # Modify execution path if needed
                if adaptations['path_modifications']:
                    self._modify_execution_path(build_id, adaptations['path_modifications'])
                    
            # Resume execution
            self._resume_build_execution(build_id)
            
        except Exception as e:
            self.logger.error(f"Failed to rebalance execution: {e}")
            self._trigger_emergency_recovery(build_id)
            
    def _monitor_execution(self, build_id: str):
        """Monitor execution with real-time intervention"""
        execution_state = self._initialize_execution_state(build_id)
        
        while self._is_active_build(build_id):
            metrics = self._collect_execution_metrics(build_id)
            
            # Check for immediate intervention needs
            if self._needs_intervention(metrics):
                self._handle_execution_intervention(build_id, metrics)
                
            # Check for performance optimization
            elif self._needs_optimization(metrics):
                self._apply_performance_optimization(build_id, metrics)
                
            # Update execution state
            self._update_execution_state(execution_state, metrics)
            
            time.sleep(self.monitoring_interval)
            
    def _needs_intervention(self, metrics: Dict) -> bool:
        """Check if execution requires immediate intervention"""
        return (
            self._is_critically_slow(metrics) or
            self._is_resource_exhausted(metrics) or
            self._has_dangerous_deviation(metrics)
        )
        
    def _handle_execution_intervention(self, build_id: str, metrics: Dict):
        """Handle critical execution issues with AI assistance"""
        # Pause execution
        self._pause_execution(build_id)
        
        try:
            # Get AI suggestions
            if hasattr(self, 'ai'):
                modifications = self.ai.suggest_execution_modification(metrics)
                if modifications and self._is_safe_modification(modifications):
                    # Apply modifications
                    success = self._apply_execution_modifications(build_id, modifications)
                    if success:
                        # Resume with modified execution
                        self._resume_execution(build_id)
                        return
                        
            # Fallback to emergency handling
            self._handle_emergency_intervention(build_id)
            
        except Exception as e:
            logging.error(f"Intervention failed: {e}")
            self._trigger_emergency_shutdown(build_id)
            
    def _apply_performance_optimizations(self, build_id: str, metrics: Dict):
        """Apply real-time performance optimizations"""
        if hasattr(self, 'ai'):
            adjustments = self.ai.suggest_realtime_adjustments(metrics)
            
            if 'resource_allocation' in adjustments:
                self._adjust_resource_allocation(build_id, adjustments['resource_allocation'])
                
            if 'execution_path' in adjustments:
                self._modify_execution_path(build_id, adjustments['execution_path'])
                
    def _rebalance_resources(self, build_id: str):
        """Dynamically rebalance system resources"""
        current_allocation = self._get_resource_allocation(build_id)
        system_load = self._get_system_metrics()
        
        # Calculate optimal allocation
        optimal = self._calculate_optimal_allocation(current_allocation, system_load)
        
        # Apply new allocation if significantly different
        if self._should_rebalance(current_allocation, optimal):
            self._apply_resource_allocation(build_id, optimal)
            
    def _monitor_system_resources(self):
        """Monitor and balance system resources across builds"""
        while self._monitoring:
            current_metrics = self.resource_monitor.get_metrics()
            active_builds = list(self.active_builds.values())
            
            if len(active_builds) > 1:
                self._balance_system_resources(active_builds, current_metrics)
                
            time.sleep(self.resource_check_interval)
            
    def _balance_system_resources(self, builds: List[Dict], metrics: Dict):
        """Balance resources across multiple builds"""
        if not hasattr(self, 'ai'):
            return
            
        try:
            # Get AI-suggested resource allocation
            allocation = self.ai.optimize_resource_allocation(builds)
            
            # Apply new allocations if significantly different
            current_allocation = self._get_current_allocation()
            if self._should_rebalance(current_allocation, allocation):
                self._apply_resource_allocation(allocation)
                
        except Exception as e:
            logging.error(f"Resource balancing failed: {e}")
            
    def _monitor_build_health(self, build_id: str):
        """Monitor build health with AI-driven intervention"""
        while build_id in self.active_builds:
            metrics = self._collect_build_metrics(build_id)
            
            # Check for potential issues
            if self._detect_potential_issues(metrics):
                self._handle_potential_issues(build_id, metrics)
                
            # Check for resource inefficiencies
            if self._detect_resource_inefficiency(metrics):
                self._optimize_resource_usage(build_id)
                
            time.sleep(self.health_check_interval)
            
    def _monitor_system_health(self):
        """Monitor system-wide health and performance"""
        while self._monitoring:
            metrics = self._collect_system_metrics()
            
            # Analyze system performance
            analysis = self.ai.analyze_system_performance(metrics)
            
            if analysis['requires_optimization']:
                self._optimize_system_resources(analysis['optimization_plan'])
                
            # Check for resource bottlenecks
            if self._detect_resource_bottlenecks(metrics):
                self._rebalance_workloads()
                
            # Monitor execution drift
            if self._detect_execution_drift(metrics):
                self._correct_execution_drift()
                
            time.sleep(self.system_check_interval)
            
    def _rebalance_workloads(self):
        """Dynamically rebalance system workloads"""
        active_builds = self._get_active_builds()
        if not active_builds:
            return
            
        # Get AI-suggested workload distribution
        distribution = self.ai.optimize_workload_distribution(
            active_builds,
            self.system_performance
        )
        
        if distribution['confidence'] > 0.85:
            self._apply_workload_distribution(distribution['allocation'])
            
    def _monitor_system_performance(self):
        """Monitor and optimize system-wide performance"""
        while self._monitoring:
            current_metrics = self._collect_system_metrics()
            
            # Analyze global performance
            analysis = self.ai.analyze_system_performance(current_metrics)
            
            if analysis['requires_optimization']:
                self._optimize_system_resources(analysis)
                
            # Check for bottlenecks
            if self._detect_resource_bottlenecks(current_metrics):
                self._rebalance_system_workload()
                
            # Monitor execution drift
            drift = self._calculate_execution_drift(current_metrics)
            if drift['significant']:
                self._correct_execution_drift(drift)
                
            time.sleep(self.system_check_interval)
            
    def _rebalance_system_workload(self):
        """Dynamically rebalance system workload"""
        active_builds = self._get_active_builds()
        system_state = self._get_system_state()
        
        # Get AI-optimized distribution
        new_distribution = self.ai.optimize_workload_distribution(
            active_builds,
            system_state
        )
        
        if new_distribution['confidence'] > 0.85:
            self._apply_workload_distribution(new_distribution)

