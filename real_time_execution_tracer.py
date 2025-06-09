#!/usr/bin/env python3
"""
Real-Time Execution Tracing System
==================================

Advanced real-time execution visualization system implementing:
- Live code execution monitoring via sys.settrace()
- Real-time visual updates with "glowing" execution paths
- Performance heatmap generation
- Self-learning pattern recognition
- Integration with 3D Visual DNA system

Implements the "watching code run" architecture from ADVANCED_VISUALIZATION_ANALYSIS.md
"""

import sys
import time
import json
import threading
import traceback
import inspect
from typing import Dict, List, Optional, Any, Callable, Tuple
from dataclasses import dataclass, field
from pathlib import Path
from collections import defaultdict, deque
import logging
import weakref

@dataclass
class ExecutionEvent:
    """Individual execution event captured by tracer"""
    timestamp: float
    file_path: str
    line_number: int
    function_name: str
    event_type: str  # 'call', 'line', 'return', 'exception'
    variables: Dict[str, Any] = field(default_factory=dict)
    execution_time: float = 0.0
    call_depth: int = 0
    thread_id: int = 0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        return {
            'timestamp': self.timestamp,
            'file_path': self.file_path,
            'line_number': self.line_number,
            'function_name': self.function_name,
            'event_type': self.event_type,
            'variables': self._serialize_variables(),
            'execution_time': self.execution_time,
            'call_depth': self.call_depth,
            'thread_id': self.thread_id
        }
    
    def _serialize_variables(self) -> Dict[str, str]:
        """Safely serialize variables for storage"""
        serialized = {}
        for key, value in self.variables.items():
            try:
                # Convert to string representation, truncate if too long
                str_value = str(value)
                if len(str_value) > 200:
                    str_value = str_value[:200] + "..."
                serialized[key] = str_value
            except Exception:
                serialized[key] = "<unserializable>"
        return serialized

@dataclass
class PerformanceHotspot:
    """Performance hotspot detected by analysis"""
    file_path: str
    function_name: str
    line_number: int
    execution_count: int
    total_time: float
    average_time: float
    max_time: float
    hotspot_score: float
    
class ExecutionPattern:
    """Execution pattern detected by self-learning system"""
    
    def __init__(self, pattern_id: str, pattern_type: str):
        self.pattern_id = pattern_id
        self.pattern_type = pattern_type  # 'bottleneck', 'hotpath', 'error_prone', 'optimization_opportunity'
        self.occurrences = 0
        self.confidence = 0.0
        self.suggestions = []
        self.impact_score = 0.0

class CodeExecutionVisualizer:
    """
    Real-time code execution visualization system
    
    Integrates with sys.settrace() to provide live monitoring and analysis
    of code execution with visual feedback and self-learning capabilities.
    """
    
    def __init__(self, workspace_path: str, max_events: int = 10000):
        self.workspace_path = Path(workspace_path)
        self.max_events = max_events
        
        # Execution tracking
        self.trace_buffer = deque(maxlen=max_events)
        self.performance_data = defaultdict(list)
        self.execution_counts = defaultdict(int)
        self.call_stack = []
        self.call_times = {}
        
        # Real-time state
        self.is_tracing = False
        self.current_hotspots = []
        self.visual_updates = {}
        self.glow_intensities = defaultdict(float)
        
        # Self-learning system
        self.learning_engine = SelfAnalysisEngine()
        self.detected_patterns = {}
        self.optimization_suggestions = []
        
        # Threading for real-time updates
        self.update_thread = None
        self.stop_event = threading.Event()
        
        # Filtering
        self.file_filters = set()
        self.function_filters = set()
        
        # Setup logging
        self.logger = self._setup_logging()
        
        # Integration points
        self.visual_dna_integration = None
        self.update_callbacks = []
        
        self.logger.info("Real-time execution visualizer initialized")
    
    def _setup_logging(self) -> logging.Logger:
        """Setup logging for execution tracer"""
        logger = logging.getLogger("ExecutionTracer")
        logger.setLevel(logging.INFO)
        
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)
        
        return logger
    
    def add_file_filter(self, file_pattern: str):
        """Add file pattern to trace (others will be ignored)"""
        self.file_filters.add(file_pattern)
    
    def add_function_filter(self, function_name: str):
        """Add function to trace (others will be ignored)"""
        self.function_filters.add(function_name)
    
    def register_update_callback(self, callback: Callable[[Dict[str, Any]], None]):
        """Register callback for real-time visual updates"""
        self.update_callbacks.append(callback)
    
    def start_tracing(self):
        """Start real-time execution tracing"""
        if self.is_tracing:
            self.logger.warning("Tracing already active")
            return
        
        self.logger.info("Starting real-time execution tracing")
        
        # Clear previous data
        self.trace_buffer.clear()
        self.performance_data.clear()
        self.execution_counts.clear()
        self.call_stack.clear()
        self.call_times.clear()
        
        # Set up tracer
        sys.settrace(self._trace_execution)
        self.is_tracing = True
        
        # Start real-time update thread
        self.stop_event.clear()
        self.update_thread = threading.Thread(target=self._real_time_update_loop)
        self.update_thread.daemon = True
        self.update_thread.start()
        
        self.logger.info("Real-time tracing started successfully")
    
    def stop_tracing(self):
        """Stop real-time execution tracing"""
        if not self.is_tracing:
            return
        
        self.logger.info("Stopping real-time execution tracing")
        
        # Stop tracer
        sys.settrace(None)
        self.is_tracing = False
        
        # Stop update thread
        self.stop_event.set()
        if self.update_thread and self.update_thread.is_alive():
            self.update_thread.join(timeout=2.0)
        
        # Final analysis
        final_analysis = self.analyze_execution_patterns()
        
        self.logger.info("Real-time tracing stopped")
        return final_analysis
    
    def _trace_execution(self, frame, event, arg):
        """Main trace function called by sys.settrace()"""
        try:
            # Get execution context
            file_path = frame.f_code.co_filename
            line_number = frame.f_lineno
            function_name = frame.f_code.co_name
            
            # Apply filters
            if self.file_filters and not any(pattern in file_path for pattern in self.file_filters):
                return
            
            if self.function_filters and function_name not in self.function_filters:
                return
            
            # Create execution event
            execution_event = ExecutionEvent(
                timestamp=time.time(),
                file_path=file_path,
                line_number=line_number,
                function_name=function_name,
                event_type=event,
                variables=dict(frame.f_locals),
                call_depth=len(self.call_stack),
                thread_id=threading.get_ident()
            )
            
            # Handle different event types
            if event == 'call':
                self._handle_call_event(execution_event, frame)
            elif event == 'line':
                self._handle_line_event(execution_event, frame)
            elif event == 'return':
                self._handle_return_event(execution_event, frame, arg)
            elif event == 'exception':
                self._handle_exception_event(execution_event, frame, arg)
            
            # Add to trace buffer
            self.trace_buffer.append(execution_event)
            
            # Update execution counts
            location_key = f"{file_path}:{function_name}:{line_number}"
            self.execution_counts[location_key] += 1
            
            # Trigger real-time updates
            self._trigger_visual_update(execution_event)
            
            return self._trace_execution  # Continue tracing
            
        except Exception as e:
            # Avoid recursive tracing errors
            if self.is_tracing:
                self.logger.error(f"Error in trace function: {e}")
    
    def _handle_call_event(self, event: ExecutionEvent, frame):
        """Handle function call event"""
        call_info = {
            'event': event,
            'start_time': event.timestamp,
            'frame_id': id(frame)
        }
        
        self.call_stack.append(call_info)
        self.call_times[id(frame)] = event.timestamp
    
    def _handle_line_event(self, event: ExecutionEvent, frame):
        """Handle line execution event"""
        # Update glow intensity for this line
        location_key = f"{event.file_path}:{event.line_number}"
        self.glow_intensities[location_key] = min(self.glow_intensities[location_key] + 0.1, 1.0)
    
    def _handle_return_event(self, event: ExecutionEvent, frame, return_value):
        """Handle function return event"""
        frame_id = id(frame)
        
        if frame_id in self.call_times:
            start_time = self.call_times[frame_id]
            execution_time = event.timestamp - start_time
            event.execution_time = execution_time
            
            # Store performance data
            function_key = f"{event.file_path}:{event.function_name}"
            self.performance_data[function_key].append(execution_time)
            
            # Clean up
            del self.call_times[frame_id]
        
        # Remove from call stack
        if self.call_stack:
            self.call_stack.pop()
    
    def _handle_exception_event(self, event: ExecutionEvent, frame, exception_info):
        """Handle exception event"""
        event.variables['exception'] = str(exception_info)
        
        # Log exception for pattern analysis
        self.learning_engine.record_exception_pattern(event)
    
    def _trigger_visual_update(self, event: ExecutionEvent):
        """Trigger real-time visual update"""
        update_data = {
            'event': event.to_dict(),
            'glow_updates': dict(self.glow_intensities),
            'current_hotspots': [h.__dict__ for h in self.current_hotspots],
            'execution_counts': dict(self.execution_counts),
            'call_stack_depth': len(self.call_stack)
        }
        
        # Call registered callbacks
        for callback in self.update_callbacks:
            try:
                callback(update_data)
            except Exception as e:
                self.logger.error(f"Error in update callback: {e}")
    
    def _real_time_update_loop(self):
        """Real-time update loop running in separate thread"""
        while not self.stop_event.is_set():
            try:
                # Update glow decay
                self._update_glow_decay()
                
                # Detect performance hotspots
                self._detect_hotspots()
                
                # Run pattern analysis
                self._analyze_patterns_realtime()
                
                # Send periodic updates
                self._send_periodic_update()
                
                # Wait before next update
                time.sleep(0.1)  # 10 FPS update rate
                
            except Exception as e:
                self.logger.error(f"Error in update loop: {e}")
    
    def _update_glow_decay(self):
        """Update glow decay for visual feedback"""
        decay_rate = 0.02  # Decay per update cycle
        
        for location_key in list(self.glow_intensities.keys()):
            self.glow_intensities[location_key] = max(
                self.glow_intensities[location_key] - decay_rate, 
                0.0
            )
            
            # Remove if too dim
            if self.glow_intensities[location_key] < 0.01:
                del self.glow_intensities[location_key]
    
    def _detect_hotspots(self):
        """Detect performance hotspots in real-time"""
        self.current_hotspots.clear()
        
        for function_key, times in self.performance_data.items():
            if len(times) >= 5:  # Minimum samples
                avg_time = sum(times) / len(times)
                max_time = max(times)
                total_time = sum(times)
                
                # Calculate hotspot score
                hotspot_score = (avg_time * len(times)) + (max_time * 0.5)
                
                if hotspot_score > 0.01:  # Threshold for hotspot
                    file_path, function_name = function_key.rsplit(':', 1)
                    
                    hotspot = PerformanceHotspot(
                        file_path=file_path,
                        function_name=function_name,
                        line_number=0,  # Would need more tracking for exact line
                        execution_count=len(times),
                        total_time=total_time,
                        average_time=avg_time,
                        max_time=max_time,
                        hotspot_score=hotspot_score
                    )
                    
                    self.current_hotspots.append(hotspot)
        
        # Sort by hotspot score
        self.current_hotspots.sort(key=lambda h: h.hotspot_score, reverse=True)
        self.current_hotspots = self.current_hotspots[:10]  # Top 10 hotspots
    
    def _analyze_patterns_realtime(self):
        """Analyze execution patterns in real-time"""
        if len(self.trace_buffer) >= 100:  # Minimum buffer for analysis
            recent_events = list(self.trace_buffer)[-100:]  # Last 100 events
            patterns = self.learning_engine.analyze_execution_patterns(recent_events)
            
            # Update detected patterns
            for pattern_id, pattern in patterns.items():
                if pattern_id not in self.detected_patterns:
                    self.detected_patterns[pattern_id] = pattern
                else:
                    # Update existing pattern
                    existing = self.detected_patterns[pattern_id]
                    existing.occurrences += pattern.occurrences
                    existing.confidence = (existing.confidence + pattern.confidence) / 2
    
    def _send_periodic_update(self):
        """Send periodic comprehensive update"""
        if self.visual_dna_integration:
            update_data = {
                'type': 'periodic_update',
                'timestamp': time.time(),
                'hotspots': [h.__dict__ for h in self.current_hotspots],
                'glow_map': dict(self.glow_intensities),
                'execution_stats': self._get_execution_stats(),
                'detected_patterns': {pid: p.__dict__ for pid, p in self.detected_patterns.items()}
            }
            
            try:
                self.visual_dna_integration.update_real_time_visualization(update_data)
            except Exception as e:
                self.logger.error(f"Error sending update to Visual DNA: {e}")
    
    def _get_execution_stats(self) -> Dict[str, Any]:
        """Get current execution statistics"""
        return {
            'total_events': len(self.trace_buffer),
            'events_per_second': len(self.trace_buffer) / max(time.time() - (self.trace_buffer[0].timestamp if self.trace_buffer else time.time()), 1),
            'call_stack_depth': len(self.call_stack),
            'unique_locations': len(self.execution_counts),
            'hotspot_count': len(self.current_hotspots),
            'pattern_count': len(self.detected_patterns)
        }
    
    def analyze_execution_patterns(self) -> Dict[str, Any]:
        """Comprehensive analysis of execution patterns"""
        self.logger.info("Analyzing execution patterns...")
        
        if not self.trace_buffer:
            return {'status': 'no_data', 'message': 'No execution data to analyze'}
        
        events_list = list(self.trace_buffer)
        
        # Comprehensive analysis
        analysis = {
            'execution_summary': self._analyze_execution_summary(events_list),
            'performance_analysis': self._analyze_performance_data(),
            'hotspot_analysis': self._analyze_hotspots(),
            'pattern_analysis': self._analyze_detected_patterns(),
            'optimization_suggestions': self._generate_optimization_suggestions(),
            'visualization_updates': self._generate_visualization_updates()
        }
        
        self.logger.info("Execution pattern analysis completed")
        return analysis
    
    def _analyze_execution_summary(self, events: List[ExecutionEvent]) -> Dict[str, Any]:
        """Analyze execution summary statistics"""
        if not events:
            return {}
        
        start_time = events[0].timestamp
        end_time = events[-1].timestamp
        duration = end_time - start_time
        
        event_types = defaultdict(int)
        files_executed = set()
        functions_executed = set()
        
        for event in events:
            event_types[event.event_type] += 1
            files_executed.add(event.file_path)
            functions_executed.add(event.function_name)
        
        return {
            'duration': duration,
            'total_events': len(events),
            'events_per_second': len(events) / max(duration, 0.001),
            'event_types': dict(event_types),
            'files_executed': len(files_executed),
            'functions_executed': len(functions_executed),
            'call_depth_max': max((e.call_depth for e in events), default=0)
        }
    
    def _analyze_performance_data(self) -> Dict[str, Any]:
        """Analyze performance data"""
        performance_analysis = {}
        
        for function_key, times in self.performance_data.items():
            if len(times) > 0:
                performance_analysis[function_key] = {
                    'call_count': len(times),
                    'total_time': sum(times),
                    'average_time': sum(times) / len(times),
                    'min_time': min(times),
                    'max_time': max(times),
                    'time_variance': self._calculate_variance(times)
                }
        
        return performance_analysis
    
    def _calculate_variance(self, values: List[float]) -> float:
        """Calculate variance of values"""
        if len(values) < 2:
            return 0.0
        
        mean = sum(values) / len(values)
        return sum((x - mean) ** 2 for x in values) / len(values)
    
    def _analyze_hotspots(self) -> Dict[str, Any]:
        """Analyze detected hotspots"""
        return {
            'hotspot_count': len(self.current_hotspots),
            'top_hotspots': [h.__dict__ for h in self.current_hotspots[:5]],
            'total_hotspot_time': sum(h.total_time for h in self.current_hotspots)
        }
    
    def _analyze_detected_patterns(self) -> Dict[str, Any]:
        """Analyze detected execution patterns"""
        pattern_summary = defaultdict(int)
        
        for pattern in self.detected_patterns.values():
            pattern_summary[pattern.pattern_type] += 1
        
        return {
            'pattern_count': len(self.detected_patterns),
            'pattern_types': dict(pattern_summary),
            'high_confidence_patterns': [
                p.__dict__ for p in self.detected_patterns.values() 
                if p.confidence > 0.8
            ]
        }
    
    def _generate_optimization_suggestions(self) -> List[Dict[str, Any]]:
        """Generate optimization suggestions based on analysis"""
        suggestions = []
        
        # Hotspot-based suggestions
        for hotspot in self.current_hotspots[:3]:  # Top 3 hotspots
            if hotspot.average_time > 0.1:  # Slow function threshold
                suggestions.append({
                    'type': 'performance_optimization',
                    'priority': 'high',
                    'location': f"{hotspot.file_path}:{hotspot.function_name}",
                    'issue': f"Function called {hotspot.execution_count} times with average time {hotspot.average_time:.3f}s",
                    'suggestion': "Consider optimizing this function or reducing call frequency",
                    'impact_estimate': hotspot.total_time
                })
        
        # Pattern-based suggestions
        for pattern in self.detected_patterns.values():
            if pattern.pattern_type == 'bottleneck' and pattern.confidence > 0.7:
                suggestions.append({
                    'type': 'bottleneck_resolution',
                    'priority': 'medium',
                    'pattern_id': pattern.pattern_id,
                    'issue': f"Bottleneck pattern detected with {pattern.confidence:.2%} confidence",
                    'suggestion': "Review execution flow and consider async operations",
                    'impact_estimate': pattern.impact_score
                })
        
        return suggestions
    
    def _generate_visualization_updates(self) -> Dict[str, Any]:
        """Generate updates for visual DNA system"""
        return {
            'glow_map': dict(self.glow_intensities),
            'hotspot_highlights': [
                {
                    'location': f"{h.file_path}:{h.function_name}",
                    'intensity': min(h.hotspot_score * 10, 1.0),
                    'color': 'red' if h.hotspot_score > 1.0 else 'orange'
                }
                for h in self.current_hotspots
            ],
            'execution_flow': [
                {
                    'location': f"{event.file_path}:{event.line_number}",
                    'timestamp': event.timestamp,
                    'intensity': 0.8
                }
                for event in list(self.trace_buffer)[-10:]  # Last 10 events
            ]
        }
    
    def export_execution_data(self, output_path: str) -> str:
        """Export execution data for analysis"""
        export_data = {
            'metadata': {
                'export_timestamp': time.time(),
                'total_events': len(self.trace_buffer),
                'workspace_path': str(self.workspace_path)
            },
            'events': [event.to_dict() for event in self.trace_buffer],
            'performance_data': dict(self.performance_data),
            'execution_counts': dict(self.execution_counts),
            'hotspots': [h.__dict__ for h in self.current_hotspots],
            'patterns': {pid: p.__dict__ for pid, p in self.detected_patterns.items()},
            'analysis': self.analyze_execution_patterns()
        }
        
        output_file = Path(output_path)
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(export_data, f, indent=2, default=str)
        
        self.logger.info(f"Execution data exported to {output_file}")
        return str(output_file)

class SelfAnalysisEngine:
    """
    Self-learning analysis engine for execution patterns
    
    Implements machine learning-like pattern recognition for code execution
    """
    
    def __init__(self):
        self.pattern_history = defaultdict(list)
        self.exception_patterns = defaultdict(int)
        self.performance_baselines = {}
        
    def analyze_execution_patterns(self, events: List[ExecutionEvent]) -> Dict[str, ExecutionPattern]:
        """Analyze execution events and detect patterns"""
        patterns = {}
        
        # Detect bottleneck patterns
        bottlenecks = self._detect_bottleneck_patterns(events)
        patterns.update(bottlenecks)
        
        # Detect hotpath patterns
        hotpaths = self._detect_hotpath_patterns(events)
        patterns.update(hotpaths)
        
        # Detect error-prone patterns
        error_patterns = self._detect_error_patterns(events)
        patterns.update(error_patterns)
        
        return patterns
    
    def _detect_bottleneck_patterns(self, events: List[ExecutionEvent]) -> Dict[str, ExecutionPattern]:
        """Detect bottleneck patterns in execution"""
        patterns = {}
        
        # Group events by function
        function_events = defaultdict(list)
        for event in events:
            if event.event_type in ['call', 'return']:
                function_key = f"{event.file_path}:{event.function_name}"
                function_events[function_key].append(event)
        
        # Analyze each function for bottleneck patterns
        for function_key, func_events in function_events.items():
            if len(func_events) >= 4:  # Minimum events for pattern
                # Calculate time spent
                call_times = []
                for i in range(0, len(func_events) - 1, 2):
                    if (i + 1 < len(func_events) and 
                        func_events[i].event_type == 'call' and 
                        func_events[i + 1].event_type == 'return'):
                        
                        duration = func_events[i + 1].timestamp - func_events[i].timestamp
                        call_times.append(duration)
                
                if call_times and len(call_times) >= 2:
                    avg_time = sum(call_times) / len(call_times)
                    
                    if avg_time > 0.05:  # 50ms threshold for bottleneck
                        pattern = ExecutionPattern(
                            pattern_id=f"bottleneck_{function_key}",
                            pattern_type="bottleneck"
                        )
                        pattern.occurrences = len(call_times)
                        pattern.confidence = min(avg_time * 10, 1.0)
                        pattern.impact_score = avg_time * len(call_times)
                        pattern.suggestions = [
                            f"Function {function_key} takes {avg_time:.3f}s on average",
                            "Consider optimization or caching"
                        ]
                        
                        patterns[pattern.pattern_id] = pattern
        
        return patterns
    
    def _detect_hotpath_patterns(self, events: List[ExecutionEvent]) -> Dict[str, ExecutionPattern]:
        """Detect frequently executed code paths"""
        patterns = {}
        
        # Count execution frequency
        location_counts = defaultdict(int)
        for event in events:
            if event.event_type == 'line':
                location_key = f"{event.file_path}:{event.line_number}"
                location_counts[location_key] += 1
        
        # Find hot paths
        for location_key, count in location_counts.items():
            if count > 50:  # Threshold for hot path
                pattern = ExecutionPattern(
                    pattern_id=f"hotpath_{location_key}",
                    pattern_type="hotpath"
                )
                pattern.occurrences = count
                pattern.confidence = min(count / 100.0, 1.0)
                pattern.impact_score = count * 0.1
                pattern.suggestions = [
                    f"Line {location_key} executed {count} times",
                    "Consider loop optimization or code restructuring"
                ]
                
                patterns[pattern.pattern_id] = pattern
        
        return patterns
    
    def _detect_error_patterns(self, events: List[ExecutionEvent]) -> Dict[str, ExecutionPattern]:
        """Detect error-prone execution patterns"""
        patterns = {}
        
        # Find exception events
        exception_events = [e for e in events if e.event_type == 'exception']
        
        if exception_events:
            # Group by location
            exception_locations = defaultdict(int)
            for event in exception_events:
                location_key = f"{event.file_path}:{event.function_name}"
                exception_locations[location_key] += 1
            
            for location_key, count in exception_locations.items():
                pattern = ExecutionPattern(
                    pattern_id=f"error_prone_{location_key}",
                    pattern_type="error_prone"
                )
                pattern.occurrences = count
                pattern.confidence = min(count / 5.0, 1.0)
                pattern.impact_score = count * 2.0  # High impact for errors
                pattern.suggestions = [
                    f"Function {location_key} threw {count} exceptions",
                    "Add error handling and input validation"
                ]
                
                patterns[pattern.pattern_id] = pattern
        
        return patterns
    
    def record_exception_pattern(self, event: ExecutionEvent):
        """Record exception pattern for learning"""
        location_key = f"{event.file_path}:{event.function_name}"
        self.exception_patterns[location_key] += 1

if __name__ == "__main__":
    # Example usage
    workspace = r"C:\Users\lokee\Documents\fake_singularity"
    
    # Initialize execution visualizer
    visualizer = CodeExecutionVisualizer(workspace)
    
    # Add filters to focus on specific files
    visualizer.add_file_filter("fake_singularity")
    
    # Register update callback
    def visual_update_callback(update_data):
        print(f"Visual update: {len(update_data['glow_updates'])} glowing locations")
    
    visualizer.register_update_callback(visual_update_callback)
    
    # Start tracing
    print("Starting execution tracing...")
    visualizer.start_tracing()
    
    # Example code to trace
    try:
        def example_function():
            time.sleep(0.01)  # Simulate some work
            return sum(range(100))
        
        def another_function():
            for i in range(5):
                result = example_function()
            return result
        
        # Execute traced code
        final_result = another_function()
        print(f"Example execution result: {final_result}")
        
    finally:
        # Stop tracing and get analysis
        analysis = visualizer.stop_tracing()
        
        if analysis:
            print("\nExecution Analysis:")
            print(f"Total events: {analysis['execution_summary']['total_events']}")
            print(f"Hotspots found: {analysis['hotspot_analysis']['hotspot_count']}")
            print(f"Patterns detected: {analysis['pattern_analysis']['pattern_count']}")
            print(f"Optimization suggestions: {len(analysis['optimization_suggestions'])}")
            
            # Export data
            export_file = visualizer.export_execution_data("execution_trace.json")
            print(f"\nTrace data exported to: {export_file}")
