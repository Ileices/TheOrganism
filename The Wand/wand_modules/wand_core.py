import json
import threading
from pathlib import Path
from typing import Dict, List
import logging
import time
from queue import Queue, Empty
from .wand_plugin_manager import WandPluginManager  # Add this import

class WandCore:
    def __init__(self, config: Dict):
        self.config = config
        self.logger = logging.getLogger('WandCore')  # Add logger initialization
        self.services = []
        self.is_running = False
        self.build_queue = Queue()  # Initialize as Queue instead of list
        self.build_history = []
        self.current_build = None
        self.build_lock = threading.Lock()
        self.error_recovery_attempts = {}
        self.max_recovery_attempts = 3
        self.recovery_cooldown = 60  # seconds
        self.blacklisted_components = {}
        self.warning_threshold = 5  # Failures before blacklisting
        self.blacklist_duration = 3600  # Seconds (1 hour)
        self.component_analysis = {}
        self.temporary_modifications = {}
        self.execution_success_rates = {}
        self.autonomous_improvements = {}
        self.ai_governance = {}
        self.system_health_metrics = {}
        self.execution_patterns = {}
        self.healing_strategies = {}
        self.cross_build_dependencies = {}
        
    def initialize(self):
        """Initialize core systems and verify dependencies"""
        try:
            # 1. Load and verify configuration
            self._verify_config()
            
            # 2. Set up build environment
            self._setup_build_environment()
            
            # 3. Load plugins
            self._load_plugins()
            
            # 4. Initialize build system
            self.build_queue = Queue()
            self.build_results = {}
            
            return True
        except Exception as e:
            logging.error(f"Core initialization failed: {e}")
            return False
            
    def start_services(self):
        """Start background services and monitoring"""
        self.is_running = True
        threading.Thread(target=self._monitor_build_queue, daemon=True).start()
        
    def _load_plugins(self):
        """Load all registered plugins from the plugins directory"""
        plugins_path = Path(self.config['plugins_directory'])
        if not plugins_path.exists():
            plugins_path.mkdir(parents=True)
            
        self.plugin_manager = WandPluginManager(plugins_path)
        self.plugins = self.plugin_manager.load_plugins()

    def _setup_build_environment(self):
        """Configure build environment and verify paths"""
        required_dirs = ['builds', 'logs', 'plugins', 'temp']
        for dir_name in required_dirs:
            path = Path(self.config.get(f'{dir_name}_directory', dir_name))
            path.mkdir(parents=True, exist_ok=True)

    def _verify_config(self):
        """Verify all required configuration settings"""
        required_settings = ['build_directory', 'plugins_directory', 'temp_directory']
        missing = [s for s in required_settings if s not in self.config]
        if missing:
            raise ValueError(f"Missing required config settings: {', '.join(missing)}")
    
    def _monitor_build_queue(self):
        """Enhanced build queue monitoring with error recovery"""
        while self.is_running:
            try:
                # Get next build job with timeout
                try:
                    build_job = self.build_queue.get(timeout=1.0)
                except Empty:
                    continue
                
                # Process build job with recovery
                success = self._process_build_job(build_job)
                
                if success:
                    # Clear recovery attempts on success
                    step_number = build_job.get('step_number')
                    if step_number in self.error_recovery_attempts:
                        del self.error_recovery_attempts[step_number]
                        
            except Exception as e:
                # Fix error logging format
                self.logger.error(f"Build queue monitor error: {str(e)}")  # Use standard logging
                time.sleep(1)
                
    def _process_build_job(self, build_job: Dict) -> bool:
        """Process build job with AI-driven optimization"""
        # Get AI safety analysis and optimization suggestions
        if hasattr(self, 'ai'):
            optimization_plan = self.ai.optimize_execution_plan(build_job)
            
            # Apply preemptive fixes if suggested
            if optimization_plan['preemptive_fixes']:
                self._apply_preemptive_fixes(build_job, optimization_plan['preemptive_fixes'])
                
            # Optimize resource allocation
            self._configure_build_resources(optimization_plan['resource_allocation'])
            
            # Reorder tasks if optimization suggests
            if optimization_plan['optimized_task_order']:
                build_job['tasks'] = optimization_plan['optimized_task_order']
                
        # Execute build with monitoring
        return self._execute_monitored_build(build_job)
        
    def _execute_monitored_build(self, build_job: Dict) -> bool:
        """Execute build with real-time monitoring and adjustment"""
        build_id = f"build_{build_job['step_number']}_{int(time.time())}"
        
        try:
            # Start monitoring thread
            monitor_thread = threading.Thread(
                target=self.monitor._monitor_build_execution,
                args=(build_id, build_job)
            )
            monitor_thread.start()
            
            # Execute build
            success = self.builder.build_from_json(build_job)
            
            if not success and hasattr(self, 'ai'):
                # Get recovery strategy
                recovery = self.ai.suggest_recovery_strategy(build_job)
                if recovery and recovery['confidence'] > 0.7:
                    return self._attempt_recovery_execution(build_job, recovery)
                    
            return success
            
        finally:
            self._cleanup_build_execution(build_id)
            
    def _apply_preemptive_fixes(self, build_job: Dict, fixes: List[Dict]):
        """Apply AI-suggested preemptive fixes"""
        for fix in fixes:
            fix_type = fix['type']
            if fix_type == 'configuration':
                self._apply_configuration_fix(build_job, fix)
            elif fix_type == 'resource':
                self._apply_resource_fix(build_job, fix)
            elif fix_type == 'dependency':
                self._apply_dependency_fix(build_job, fix)
                
        self.logger.info(f"Applied {len(fixes)} preemptive fixes to build {build_job['step_number']}")
        
    def _check_blacklist(self, build_job: Dict) -> bool:
        """Enhanced blacklist checking with AI-driven recovery options"""
        components = self._extract_build_components(build_job)
        
        for component in components:
            status = self._get_component_status(component)
            
            if status['blacklisted']:
                # Check for AI recovery possibility
                if hasattr(self, 'ai'):
                    recovery_option = self.ai.analyze_recovery_possibility(
                        component, 
                        status['blacklist_reason']
                    )
                    
                    if recovery_option['can_recover']:
                        return self._attempt_component_recovery(component, recovery_option)
                        
                return True  # Keep blacklisted if no recovery possible
                
        return False
        
    def _attempt_component_recovery(self, component: str, recovery_option: Dict) -> bool:
        """Attempt to recover a blacklisted component"""
        try:
            # Apply AI-suggested modifications
            self._apply_component_modifications(component, recovery_option['modifications'])
            
            # Monitor recovery attempt
            success = self._monitor_component_recovery(component, 
                                                     recovery_option['monitoring_rules'])
            
            if success:
                self._remove_from_blacklist(component)
                return False  # Allow component to be used
                
            return True  # Keep blacklisted if recovery fails
            
        except Exception as e:
            logging.error(f"Component recovery failed: {e}")
            return True
            
    def _apply_component_modifications(self, component: str, modifications: Dict):
        """Apply AI-suggested component modifications"""
        for mod in modifications:
            if mod['type'] == 'configuration':
                self._update_component_config(component, mod['changes'])
            elif mod['type'] == 'execution':
                self._modify_execution_parameters(component, mod['parameters'])
            elif mod['type'] == 'dependencies':
                self._update_component_dependencies(component, mod['updates'])
                
    def _should_blacklist_component(self, component: str) -> bool:
        """AI-driven decision for component blacklisting"""
        if component not in self.component_analysis:
            self._initialize_component_analysis(component)
            
        analysis = self.component_analysis[component]
        
        # Check if AI suggests permanent blacklisting
        if hasattr(self, 'ai'):
            blacklist_decision = self.ai.analyze_blacklist_decision(component, analysis)
            return blacklist_decision.get('should_blacklist', False)
            
        # Fallback to standard threshold check
        return analysis['failure_count'] >= self.warning_threshold
        
    def _apply_temporary_modifications(self, build_job: Dict, strategy: Dict):
        """Apply AI-suggested temporary modifications"""
        job_id = build_job.get('step_number', str(time.time()))
        
        self.temporary_modifications[job_id] = {
            'original': build_job.copy(),
            'modifications': strategy['modifications'],
            'expiry': time.time() + strategy.get('duration', 3600)
        }
        
        # Apply modifications
        for mod in strategy['modifications']:
            if mod['type'] == 'parameter':
                self._modify_build_parameters(build_job, mod['changes'])
            elif mod['type'] == 'resource':
                self._modify_resource_allocation(build_job, mod['changes'])
            elif mod['type'] == 'execution':
                self._modify_execution_strategy(build_job, mod['changes'])
                
        self.logger.info(f"Applied temporary modifications to build {job_id}")
        
    def _maybe_blacklist_components(self, build_job: Dict, safety_check: Dict):
        """Consider blacklisting components based on failure patterns"""
        components = self._extract_build_components(build_job)
        current_time = time.time()
        
        for component in components:
            if component not in self.blacklisted_components:
                failure_count = self._get_component_failure_count(component)
                if (failure_count >= self.warning_threshold):
                    self.blacklisted_components[component] = {
                        'timestamp': current_time,
                        'reason': f"Exceeded failure threshold: {failure_count} failures",
                        'safety_check': safety_check
                    }
                    self.logger.warning(f"Blacklisted component: {component}")
                
    def _handle_build_failure(self, build_job: Dict):
        """Handle build failures with recovery logic"""
        step_number = build_job.get('step_number')
        current_time = time.time()
        
        # Initialize or update recovery attempts
        if step_number not in self.error_recovery_attempts:
            self.error_recovery_attempts[step_number] = {
                'count': 0,
                'last_attempt': current_time
            }
            
        recovery = self.error_recovery_attempts[step_number]
        
        # Check if we should attempt recovery
        if (recovery['count'] < self.max_recovery_attempts and 
            current_time - recovery['last_attempt'] >= self.recovery_cooldown):
            
            recovery['count'] += 1
            recovery['last_attempt'] = current_time
            
            # Requeue the build job for retry
            self.logger.log_info(f"Retrying build step {step_number} "
                               f"(attempt {recovery['count']}/{self.max_recovery_attempts})")
            self.build_queue.put(build_job)
        else:
            self.logger.log_error('core', f"Build step {step_number} failed permanently "
                                f"after {recovery['count']} attempts")
                                 
    def _execute_build(self, build_job: Dict) -> bool:
        """Execute build with autonomous AI governance"""
        build_id = self._generate_build_id(build_job)
        execution_context = self._create_execution_context(build_id)
        
        try:
            # Initialize performance tracking
            self._initialize_performance_tracking(build_id)
            
            # Execute with continuous optimization
            success = self._execute_with_optimization(build_job, execution_context)
            
            # Update success patterns
            self._update_execution_patterns(build_id, execution_context, success)
            
            if not success:
                # Attempt self-healing recovery
                recovery_success = self._attempt_self_healing(build_job, execution_context)
                if recovery_success:
                    # Store successful recovery pattern
                    self._store_healing_pattern(build_id, execution_context)
                    return True
                    
            return success
            
        finally:
            self._store_execution_metrics(build_id, execution_context)
            
    def _execute_with_optimization(self, build_job: Dict, context: Dict) -> bool:
        """Execute build with continuous AI optimization"""
        while context['attempts'] < self.max_attempts:
            try:
                # Get AI-optimized execution plan
                execution_plan = self.ai.optimize_execution_plan(build_job)
                
                # Execute with continuous monitoring
                with self._managed_execution(context) as executor:
                    while not executor.is_complete():
                        metrics = executor.get_current_metrics()
                        
                        # Check for needed adaptations
                        adaptations = self.ai.adapt_to_system_load(context['id'], metrics)
                        if adaptations['confidence'] > 0.8:
                            executor.apply_adaptations(adaptations)
                            
                    success = executor.get_result()
                    if success:
                        return True
                        
                # Get next optimization if needed
                next_strategy = self.ai.suggest_next_strategy(context)
                if not next_strategy:
                    break
                    
                # Apply optimizations and continue
                self._apply_execution_optimizations(build_job, next_strategy)
                context['attempts'] += 1
                
            except Exception as e:
                self.logger.error(f"Execution optimization failed: {e}")
                break
                
        return False
        
    def _execute_with_governance(self, build_job: Dict) -> bool:
        """Execute with AI governance and real-time optimization"""
        build_id = build_job['id']
        context = self._create_execution_context(build_id)
        
        while context['attempts'] < self.max_attempts:
            try:
                # Get optimized execution plan
                execution_plan = self.ai.optimize_execution_plan(build_job)
                
                # Execute with continuous monitoring
                with self._managed_execution(context) as executor:
                    while not executor.is_complete():
                        # Get real-time adaptations
                        adaptations = self.ai.adapt_execution_path(
                            build_id,
                            executor.get_metrics()
                        )
                        
                        if adaptations['modifications']:
                            executor.apply_modifications(adaptations['modifications'])
                            
                        if not adaptations['continue_execution']:
                            break
                            
                    success = executor.get_result()
                    if success:
                        return True
                        
                # Get next optimization if needed
                next_strategy = self.ai.suggest_next_strategy(context)
                if not next_strategy:
                    break
                    
                # Apply optimizations and continue
                self._apply_execution_optimizations(build_job, next_strategy)
                context['attempts'] += 1
                
            except Exception as e:
                self.logger.error(f"Execution failed: {e}")
                break
                
        return False
        
    def _attempt_self_healing(self, build_job: Dict) -> bool:
        """Attempt self-healing recovery with AI analysis"""
        build_id = build_job['id']
        
        # Analyze failure
        analysis = self.ai.analyze_failure_pattern(build_job)
        if not analysis:
            return False
            
        # Generate healing strategy
        healing_strategy = self._generate_healing_strategy(analysis)
        if not healing_strategy:
            return False
            
        # Apply healing modifications
        modified_job = self._apply_healing_strategy(build_job, healing_strategy)
        
        # Retry with healing
        return self._execute_build(modified_job)
        
    def _execute_with_ai_control(self, build_job: Dict, context: Dict) -> bool:
        """Execute build under AI control"""
        while context['attempts'] < self.max_attempts:
            try:
                # Get AI-optimized execution plan
                execution_plan = self.ai.optimize_execution_plan(build_job)
                
                # Start execution with continuous monitoring
                with self._managed_execution(context) as executor:
                    success = executor.execute_plan(execution_plan)
                    
                    if success:
                        return True
                        
                # Get AI suggestions for next attempt
                next_strategy = self.ai.suggest_next_strategy(context)
                if not next_strategy:
                    break
                    
                # Apply AI-suggested modifications
                self._apply_execution_modifications(build_job, next_strategy)
                context['attempts'] += 1
                
            except Exception as e:
                self.logger.error(f"AI-controlled execution failed: {e}")
                break
                
        return False
        
    def _managed_execution(self, context: Dict):
        """Context manager for AI-managed execution"""
        class ExecutionManager:
            def __init__(self, core, context):
                self.core = core
                self.context = context
                self.paused = False
                
            def __enter__(self):
                self.core._start_execution_monitoring(self.context)
                return self
                
            def __exit__(self, exc_type, exc_val, exc_tb):
                self.core._stop_execution_monitoring(self.context)
                
            def pause_execution(self):
                """Pause execution for AI intervention"""
                self.paused = True
                self.core._pause_execution(self.context)
                
            def resume_execution(self):
                """Resume execution after AI intervention"""
                self.paused = False
                self.core._resume_execution(self.context)
                
            def execute_plan(self, plan: Dict) -> bool:
                """Execute plan with AI monitoring"""
                return self.core._execute_plan_with_monitoring(plan, self.context)
                
        return ExecutionManager(self, context)
        
    def _execute_with_adaptation(self, build_job: Dict, context: Dict) -> bool:
        """Execute build with real-time adaptation"""
        while context['attempts'] < self.max_adaptation_attempts:
            try:
                # Get current execution plan
                current_plan = self._get_current_execution_plan(context)
                
                # Execute with monitoring
                success = self._execute_plan(current_plan, context)
                if success:
                    return True
                        
                # Get adaptation strategy
                adaptation = self._get_adaptation_strategy(context)
                if not adaptation:
                    break
                    
                # Apply adaptation and continue
                self._apply_adaptation(adaptation, context)
                context['attempts'] += 1
                
            except Exception as e:
                self.logger.error(f"Execution adaptation failed: {e}")
                break
                
        return False
        
    def _attempt_autonomous_recovery(self, build_job: Dict, context: Dict) -> bool:
        """Attempt autonomous recovery with AI-generated fixes"""
        # Get failure analysis
        failure_analysis = self.ai.analyze_execution_failure(context)
        
        # Generate recovery options
        recovery_options = self._generate_recovery_options(failure_analysis)
        
        for option in recovery_options:
            try:
                # Apply recovery option
                modified_job = self._apply_recovery_option(build_job, option)
                
                # Execute with recovery
                if self._execute_recovery_attempt(modified_job, context):
                    # Store successful recovery pattern
                    self._store_recovery_pattern(context, option)
                    return True
                        
            except Exception as e:
                self.logger.error(f"Recovery attempt failed: {e}")
                continue
                
        return False
        
    def _execute_with_ai_governance(self, build_job: Dict) -> bool:
        """Execute with full AI autonomy and governance"""
        build_id = build_job['id']
        context = self._create_execution_context(build_id)
        
        try:
            # Initialize AI governance
            self._initialize_ai_governance(build_id)
            
            while context['attempts'] < self.max_attempts:
                # Get AI-optimized execution plan
                execution_plan = self.ai.optimize_execution_plan(build_job)
                
                # Execute with continuous AI monitoring
                with self._managed_execution(context) as executor:
                    while not executor.is_complete():
                        # Get real-time system analysis
                        analysis = self.ai.analyze_system_state(executor.get_metrics())
                        
                        if analysis['requires_intervention']:
                            # Allow AI to make autonomous decisions
                            self._handle_ai_intervention(build_id, analysis)
                            
                        # Check for system health issues
                        health_check = self._check_system_health()
                        if not health_check['healthy']:
                            self._handle_system_health_issues(health_check)
                            
                    success = executor.get_result()
                    if success:
                        # Update AI learning state
                        self._update_ai_learning(context, success=True)
                        return True
                        
                # Get next strategy if needed
                next_strategy = self.ai.suggest_next_strategy(context)
                if not next_strategy:
                    break
                    
                self._apply_ai_strategy(build_job, next_strategy)
                context['attempts'] += 1
                
        except Exception as e:
            self.logger.error(f"AI governance failed: {e}")
            self._handle_governance_failure(build_id, e)
            
        return False
        
    def _handle_ai_intervention(self, build_id: str, analysis: Dict):
        """Handle autonomous AI intervention"""
        try:
            # Apply system optimizations
            if analysis['system_optimizations']:
                self._apply_system_optimizations(analysis['system_optimizations'])
                
            # Modify execution parameters
            if analysis['parameter_adjustments']:
                self._apply_parameter_adjustments(build_id, analysis['parameter_adjustments'])
                
            # Update execution framework
            if analysis['framework_updates']:
                self._update_execution_framework(analysis['framework_updates'])
                
        except Exception as e:
            self.logger.error(f"AI intervention failed: {e}")
            self._rollback_intervention(build_id)
            
    def _execute_with_pattern_optimization(self, build_job: Dict) -> bool:
        """Execute build with pattern-based optimization"""
        build_id = build_job['id']
        
        # Analyze historical patterns
        patterns = self._analyze_execution_patterns(build_job)
        
        if patterns['optimizations']:
            # Apply pattern-based improvements
            self._apply_pattern_optimizations(build_job, patterns['optimizations'])
            
        try:
            # Execute with continuous monitoring
            with self._managed_execution(build_id) as executor:
                while not executor.is_complete():
                    # Check for recurring issues
                    issues = self._detect_recurring_issues(executor.get_metrics())
                    if issues:
                        self._apply_proactive_fixes(build_id, issues)
                        
                    # Analyze cross-build dependencies
                    dependencies = self._analyze_cross_build_dependencies(build_id)
                    if dependencies['requires_adjustment']:
                        self._adjust_execution_sequence(build_id, dependencies)
                        
                return executor.get_result()
                
        except Exception as e:
            self.logger.error(f"Pattern-optimized execution failed: {e}")
            return False
            
    def _analyze_cross_build_dependencies(self, build_id: str) -> Dict:
        """Analyze and optimize cross-build dependencies"""
        active_builds = self._get_active_builds()
        dependency_graph = self._build_dependency_graph(active_builds)
        
        # Get AI-suggested optimizations
        optimizations = self.ai.optimize_dependency_execution(
            build_id,
            dependency_graph,
            self.execution_patterns
        )
        
        return {
            'requires_adjustment': optimizations['confidence'] > 0.8,
            'suggested_sequence': optimizations['execution_sequence'],
            'priority_adjustments': optimizations['priority_changes']
        }
        
    def _execute_build_with_patterns(self, build_job: Dict) -> bool:
        """Execute build with pattern-based optimization"""
        build_id = build_job['id']
        
        # Analyze execution patterns
        patterns = self._analyze_execution_patterns(build_job)
        
        if patterns['optimizations']:
            self._apply_pattern_optimizations(build_job, patterns['optimizations'])
            
        try:
            with self._managed_execution(build_id) as executor:
                while not executor.is_complete():
                    # Check for recurring issues
                    issues = self._detect_recurring_issues(executor.get_metrics())
                    if issues:
                        self._apply_proactive_fixes(build_id, issues)
                        
                    # Analyze cross-build dependencies
                    dependencies = self._analyze_cross_dependencies(build_id)
                    if dependencies['requires_adjustment']:
                        self._adjust_execution_sequence(build_id, dependencies)
                        
                return executor.get_result()
                
        except Exception as e:
            self.logger.error(f"Pattern-optimized execution failed: {e}")
            return False
            
    def _analyze_cross_dependencies(self, build_id: str) -> Dict:
        """Analyze and optimize cross-build dependencies"""
        active_builds = self._get_active_builds()
        dependency_graph = self._build_dependency_graph(active_builds)
        
        return self.ai.optimize_dependency_execution(
            build_id,
            dependency_graph,
            self.execution_patterns
        )
