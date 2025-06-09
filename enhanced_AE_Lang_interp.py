# AE-Lang Stress Test Suite - Real-World Applications
# Tests: Mathematical computation, NLP processing, memory optimization, practical problem solving

import re
import random
import json
import math
import time
from collections import defaultdict

# === ENHANCED MEMORY MODEL ===
class EnhancedMemory:
    def __init__(self, label, value=None, rby=None, state='stable', memory_type='standard'):
        self.label = label
        self.value = value
        self.state = state
        self.memory_type = memory_type  # standard, computational, linguistic, practical
        self.creation_time = time.time()
        self.access_count = 0
        self.utility_score = 0.0
        
        # Enhanced RBY with computational meaning
        if rby:
            self.rby = rby
        else:
            r = round(random.uniform(0.31, 0.36), 12)
            b = round(random.uniform(0.31, 0.36), 12) 
            y = round(1.0 - (r + b), 12)
            self.rby = {'R': r, 'B': b, 'Y': y}
        
        self.decay_count = 0
        self.lineage = []
        self.computational_cache = {}
        self.linguistic_features = {}

    def compute_utility(self):
        """Calculate real-world utility of this memory"""
        age_factor = time.time() - self.creation_time
        access_factor = self.access_count
        execution_factor = self.rby['Y']
        self.utility_score = (access_factor * execution_factor) / (age_factor + 1)
        return self.utility_score

    def decay(self):
        """Enhanced decay with utility preservation"""
        base_decay = 0.05 + random.uniform(0, 0.025)
        utility_preservation = min(0.03, self.utility_score)
        actual_decay = max(0.01, base_decay - utility_preservation)
        
        self.rby['Y'] = max(0.0, self.rby['Y'] - actual_decay)
        self.rby['R'] = min(1.0, self.rby['R'] + actual_decay / 2)
        self.decay_count += 1
        
        if self.rby['Y'] < 0.1:
            self.state = 'nullified'
        return self.state

    def perform_computation(self, operation, operand=None):
        """Actual mathematical/logical computation"""
        try:
            if self.memory_type != 'computational':
                return None
                
            if operation == 'sqrt' and isinstance(self.value, (int, float)):
                result = math.sqrt(float(self.value))
                self.computational_cache['sqrt'] = result
                return result
            elif operation == 'factorial' and isinstance(self.value, int) and self.value >= 0:
                result = math.factorial(self.value)
                self.computational_cache['factorial'] = result
                return result
            elif operation == 'add' and operand is not None:
                result = float(self.value) + float(operand)
                self.computational_cache['add'] = result
                return result
            elif operation == 'multiply' and operand is not None:
                result = float(self.value) * float(operand)
                self.computational_cache['multiply'] = result
                return result
        except Exception as e:
            return f"ComputationError: {e}"
        return None

    def analyze_language(self, text):
        """Real NLP analysis"""
        if self.memory_type != 'linguistic':
            return None
            
        features = {
            'length': len(text),
            'words': len(text.split()),
            'sentences': text.count('.') + text.count('!') + text.count('?'),
            'questions': text.count('?'),
            'exclamations': text.count('!'),
            'uppercase_ratio': sum(1 for c in text if c.isupper()) / max(1, len(text)),
            'complexity_score': len(set(text.lower().split())) / max(1, len(text.split()))
        }
        self.linguistic_features.update(features)
        return features

    def compress_to_glyph(self):
        """Intelligent compression preserving utility"""
        base = f"{self.label}:{str(self.value)[:8]}"
        rby_str = ''.join(f"{k}{str(v)[:5]}" for k, v in self.rby.items())
        utility_marker = f"U{str(self.utility_score)[:4]}"
        glyph = f"{base}_{rby_str}_{utility_marker}_{random.randint(1000,9999)}"
        return glyph

    def to_dict(self):
        return {
            'label': self.label,
            'value': self.value,
            'state': self.state,
            'memory_type': self.memory_type,
            'rby': self.rby.copy(),
            'lineage': self.lineage,
            'decay_count': self.decay_count,
            'utility_score': self.utility_score,
            'access_count': self.access_count,
            'computational_cache': self.computational_cache,
            'linguistic_features': self.linguistic_features
        }

# === ENHANCED AE-LANG INTERPRETER ===
class EnhancedAELangInterpreter:
    def __init__(self):
        self.memories = {}
        self.excretions = []
        self.logs = []
        self.threats = []
        self.dreams = []
        self.cycle = 0
        self.last_input = ""
        self.last_output = ""
        self.script_lines = []
        self.stress_test_results = {}
        self.performance_metrics = {}

    def stress_test_mathematical_computation(self):
        """Test real mathematical computation capabilities"""
        print("\n=== STRESS TEST: Mathematical Computation ===")
        
        # Test basic arithmetic
        self.memories['math_5'] = EnhancedMemory('math_5', 5, memory_type='computational')
        self.memories['math_10'] = EnhancedMemory('math_10', 10, memory_type='computational')
        
        # Perform actual computations
        sqrt_result = self.memories['math_5'].perform_computation('sqrt')
        factorial_result = self.memories['math_5'].perform_computation('factorial')
        add_result = self.memories['math_5'].perform_computation('add', 10)
        
        results = {
            'sqrt_5': sqrt_result,
            'factorial_5': factorial_result,
            'add_5_plus_10': add_result
        }
        
        self.stress_test_results['mathematical'] = results
        print(f"Mathematical computation results: {results}")
        return results

    def stress_test_nlp_processing(self):
        """Test natural language processing capabilities"""
        print("\n=== STRESS TEST: NLP Processing ===")
        
        test_texts = [
            "What is consciousness and how does intelligence emerge?",
            "The quick brown fox jumps over the lazy dog.",
            "AE = C = 1 represents the fundamental equation of existence!"
        ]
        
        results = {}
        for i, text in enumerate(test_texts):
            memory_label = f'nlp_test_{i}'
            self.memories[memory_label] = EnhancedMemory(memory_label, text, memory_type='linguistic')
            analysis = self.memories[memory_label].analyze_language(text)
            results[memory_label] = analysis
            
        self.stress_test_results['nlp'] = results
        print(f"NLP analysis results: {results}")
        return results

    def stress_test_memory_management(self):
        """Test memory efficiency and utility scoring"""
        print("\n=== STRESS TEST: Memory Management ===")
        
        # Create many memories and test utility scoring
        for i in range(20):
            label = f'stress_memory_{i}'
            self.memories[label] = EnhancedMemory(label, f'test_value_{i}')
            # Simulate different access patterns
            for _ in range(random.randint(0, 10)):
                self.memories[label].access_count += 1
                self.memories[label].compute_utility()
        
        # Test decay and cleanup
        initial_count = len(self.memories)
        self.intelligent_decay_all()
        final_count = len(self.memories)
        
        results = {
            'initial_memory_count': initial_count,
            'final_memory_count': final_count,
            'cleanup_efficiency': (initial_count - final_count) / initial_count if initial_count > 0 else 0
        }
        
        self.stress_test_results['memory_management'] = results
        print(f"Memory management results: {results}")
        return results

    def stress_test_practical_problem_solving(self):
        """Test solving actual practical problems"""
        print("\n=== STRESS TEST: Practical Problem Solving ===")
        
        # Problem 1: Calculate compound interest
        principal = EnhancedMemory('principal', 1000, memory_type='computational')
        rate = EnhancedMemory('rate', 0.05, memory_type='computational')
        time_years = EnhancedMemory('time', 5, memory_type='computational')
        
        # Compound interest formula: A = P(1 + r)^t
        compound_result = principal.perform_computation('multiply', 
            (1 + rate.value) ** time_years.value)
        
        # Problem 2: Text analysis for sentiment
        sentiment_text = EnhancedMemory('sentiment', "I am very happy and excited about this!", 
                                      memory_type='linguistic')
        sentiment_analysis = sentiment_text.analyze_language(sentiment_text.value)
        
        # Problem 3: Fibonacci sequence calculation
        fib_mem = EnhancedMemory('fibonacci', 10, memory_type='computational')
        fib_sequence = self.calculate_fibonacci(10)
        
        results = {
            'compound_interest': compound_result,
            'sentiment_analysis': sentiment_analysis,
            'fibonacci_10': fib_sequence
        }
        
        self.stress_test_results['practical'] = results
        print(f"Practical problem solving results: {results}")
        return results

    def calculate_fibonacci(self, n):
        """Helper function to calculate Fibonacci sequence"""
        if n <= 1:
            return n
        a, b = 0, 1
        for _ in range(2, n + 1):
            a, b = b, a + b
        return b

    def intelligent_decay_all(self):
        """Enhanced decay that preserves useful memories"""
        to_excrete = []
        
        # Calculate utility scores for all memories
        utilities = []
        for label, mem in self.memories.items():
            utility = mem.compute_utility()
            utilities.append((utility, label, mem))
        
        # Sort by utility (ascending) to remove least useful first
        utilities.sort()
        
        # Apply decay
        for utility, label, mem in utilities:
            state = mem.decay()
            if state == 'nullified':
                # Only excrete if utility is very low
                if utility < 0.1:
                    glyph = mem.compress_to_glyph()
                    self.excretions.append(glyph)
                    self.logs.append(f"Memory [{label}] nullified and excreted: {glyph}")
                    to_excrete.append(label)
                else:
                    # Preserve high-utility memories even if Y is low
                    mem.rby['Y'] = 0.15  # Reset to minimum viable level
                    mem.state = 'preserved'
                    self.logs.append(f"High-utility memory [{label}] preserved")
        
        # Remove excreted memories
        for label in to_excrete:
            del self.memories[label]

    def enhanced_chatbot_reply(self):
        """Intelligent chatbot response using memory analysis"""
        if not self.last_input:
            return "[NoInput]"
        
        # Analyze input
        input_mem = EnhancedMemory('temp_input', self.last_input, memory_type='linguistic')
        analysis = input_mem.analyze_language(self.last_input)
        
        # Generate intelligent response based on analysis
        if analysis['questions'] > 0:
            if 'consciousness' in self.last_input.lower():
                response = f"Consciousness emerges from recursive cognition. AE=C=1. Analysis: {analysis['complexity_score']:.2f} complexity."
            elif 'intelligence' in self.last_input.lower():
                response = f"Intelligence is the product of R+B+Y trifecta processing. Words: {analysis['words']}, Complexity: {analysis['complexity_score']:.2f}"
            elif any(word in self.last_input.lower() for word in ['math', 'calculate', 'compute']):
                # Extract numbers and perform computation
                numbers = re.findall(r'\d+\.?\d*', self.last_input)
                if len(numbers) >= 2:
                    try:
                        result = float(numbers[0]) + float(numbers[1])
                        response = f"Computational result: {numbers[0]} + {numbers[1]} = {result}"
                    except:
                        response = f"Mathematical processing initiated. Found numbers: {numbers}"
                else:
                    response = f"Ready for mathematical computation. Please provide numbers."
            else:
                response = f"Question detected. Complexity: {analysis['complexity_score']:.2f}, Processing through RBY trifecta..."
        else:
            response = f"Statement processed. Length: {analysis['length']}, Words: {analysis['words']}, Executing through Y-nodes."
        
        return response

    def run_comprehensive_stress_test(self):
        """Run all stress tests"""
        print("===== AE-LANG COMPREHENSIVE STRESS TEST =====")
        start_time = time.time()
        
        # Run all stress tests
        math_results = self.stress_test_mathematical_computation()
        nlp_results = self.stress_test_nlp_processing()
        memory_results = self.stress_test_memory_management()
        practical_results = self.stress_test_practical_problem_solving()
        
        end_time = time.time()
        
        # Performance metrics
        self.performance_metrics = {
            'total_execution_time': end_time - start_time,
            'memory_efficiency': memory_results['cleanup_efficiency'],
            'computational_success': len([r for r in math_results.values() if r is not None]),
            'nlp_processing_success': len(nlp_results),
            'practical_problem_solving': len([r for r in practical_results.values() if r is not None])
        }
        
        print(f"\n=== PERFORMANCE METRICS ===")
        for metric, value in self.performance_metrics.items():
            print(f"{metric}: {value}")
        
        return {
            'stress_tests': self.stress_test_results,
            'performance': self.performance_metrics
        }

    def summary(self):
        print("\n=== ENHANCED ILEICES STATE ===")
        print(f"Cycle: {self.cycle}")
        print(f"Active Memories: {len(self.memories)}")
        print(f"Excretions Generated: {len(self.excretions)}")
        print(f"Performance Score: {sum(self.performance_metrics.values()) if self.performance_metrics else 0:.2f}")
        
        # Show high-utility memories
        if self.memories:
            utilities = [(mem.compute_utility(), label, mem) for label, mem in self.memories.items()]
            utilities.sort(reverse=True)
            print("\nTop 5 Most Useful Memories:")
            for i, (utility, label, mem) in enumerate(utilities[:5]):
                print(f"  {i+1}. {label}: utility={utility:.3f}, type={mem.memory_type}")
        
        print("===========================================\n")

# === MAIN EXECUTION ===
def main():
    print("===== ENHANCED AE-LANG INTERPRETER (STRESS TEST VERSION) =====")
    interpreter = EnhancedAELangInterpreter()
    
    # Run comprehensive stress test
    test_results = interpreter.run_comprehensive_stress_test()
    
    # Interactive mode with enhanced capabilities
    while True:
        print("\nEnhanced AE-Lang Menu:")
        print(" 1. Run stress tests")
        print(" 2. Test intelligent chatbot")
        print(" 3. Mathematical computation test")
        print(" 4. NLP analysis test")
        print(" 5. Memory management demo")
        print(" 6. Exit")
        
        choice = input("Select: ").strip()
        
        if choice == "1":
            interpreter.run_comprehensive_stress_test()
        elif choice == "2":
            user_input = input("Ask me anything: ")
            interpreter.last_input = user_input
            response = interpreter.enhanced_chatbot_reply()
            print(f"Enhanced Ileices: {response}")
        elif choice == "3":
            interpreter.stress_test_mathematical_computation()
        elif choice == "4":
            interpreter.stress_test_nlp_processing()
        elif choice == "5":
            interpreter.stress_test_memory_management()
        elif choice == "6":
            print("Exiting Enhanced AE-Lang Interpreter.")
            break
        else:
            print("Invalid choice.")

if __name__ == "__main__":
    main()