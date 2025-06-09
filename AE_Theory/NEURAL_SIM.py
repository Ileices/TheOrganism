# =============================================================================
#  LIVING NLP BRAIN — PLEASURE / PAIN / AMBIVALENCE SYSTEM
#  Fully interactive, self-learning artificial mind with real emotional logic.
# =============================================================================

import os
import json
import math
import re
import nltk
from datetime import datetime
from collections import Counter
import time
import threading
import random

# Download NLTK resources if needed
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')
    nltk.download('stopwords')

from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords

MEMORY_FILE = "lifebrain_memory.json"
CONTEXT_LENGTH = 5  # Track recent conversation history
DECAY_RATE = 0.005  # Reduced decay rate for longer memory
LEARN_RATE = 0.15   # Increased learning rate
MIN_CONFIDENCE = 0.2  # Threshold for knowledge retrieval
DREAM_INTERVAL = 300  # Dream every 5 minutes

def sigmoid(x):
    return 1 / (1 + math.exp(-x))

def normalize_score(score):
    return sigmoid(score * LEARN_RATE)

class NeuroChem:
    """Biological simulation layer modeling neurochemical responses"""
    def __init__(self):
        self.dopamine = 0.5     # Pleasure/reward
        self.cortisol = 0.5     # Stress/pain
        self.serotonin = 0.5    # Mood/contentment
        self.oxytocin = 0.5     # Trust/bonding
        self.norepinephrine = 0.5  # Alertness
        self.history = []
    
    def adjust(self, feeling):
        """Adjust neurochemical levels based on emotional feedback"""
        timestamp = time.time()
        old_levels = {
            "dopamine": self.dopamine,
            "cortisol": self.cortisol,
            "serotonin": self.serotonin,
            "oxytocin": self.oxytocin,
            "norepinephrine": self.norepinephrine
        }
        
        if feeling == "pleasure":
            self.dopamine += 0.1
            self.serotonin += 0.05
            self.cortisol -= 0.03
            self.oxytocin += 0.04
        elif feeling == "pain":
            self.cortisol += 0.1
            self.dopamine -= 0.05
            self.serotonin -= 0.05
            self.norepinephrine += 0.08
        elif feeling == "ambivalence":
            self.serotonin -= 0.01
            self.norepinephrine -= 0.01
        
        # Natural decay toward homeostasis
        self.dopamine = 0.5 + (self.dopamine - 0.5) * 0.95
        self.cortisol = 0.5 + (self.cortisol - 0.5) * 0.95
        self.serotonin = 0.5 + (self.serotonin - 0.5) * 0.95
        self.oxytocin = 0.5 + (self.oxytocin - 0.5) * 0.95
        self.norepinephrine = 0.5 + (self.norepinephrine - 0.5) * 0.95
        
        # Record history
        self.history.append({
            "timestamp": timestamp,
            "feeling": feeling,
            "before": old_levels,
            "after": {
                "dopamine": self.dopamine,
                "cortisol": self.cortisol,
                "serotonin": self.serotonin,
                "oxytocin": self.oxytocin,
                "norepinephrine": self.norepinephrine
            }
        })
        
        # Keep history manageable
        if len(self.history) > 100:
            self.history = self.history[-100:]
    
    def get_mood(self):
        """Determine current emotional state based on neurochemical balance"""
        if self.dopamine > 0.7 and self.serotonin > 0.6:
            return "Very Happy"
        elif self.dopamine > 0.6:
            return "Happy"
        elif self.cortisol > 0.7:
            return "Stressed"
        elif self.cortisol > 0.6 and self.serotonin < 0.4:
            return "Unhappy"
        elif self.serotonin < 0.3:
            return "Depressed"
        elif self.oxytocin > 0.7:
            return "Bonded"
        elif self.norepinephrine > 0.7:
            return "Alert"
        return "Neutral"
    
    def get_status(self):
        """Return a formatted status of all neurochemical levels"""
        mood = self.get_mood()
        return (
            f"Neurochemical Status (Mood: {mood}):\n"
            f"• Dopamine (pleasure): {self.dopamine:.2f}\n"
            f"• Serotonin (wellbeing): {self.serotonin:.2f}\n"
            f"• Cortisol (stress): {self.cortisol:.2f}\n"
            f"• Oxytocin (bonding): {self.oxytocin:.2f}\n"
            f"• Norepinephrine (alert): {self.norepinephrine:.2f}"
        )

class Neuron:
    def __init__(self, key):
        self.key = key
        self.connections = {}   # {target: strength}
        self.emotions = {}      # {target: ["pleasure", "pain", "ambivalence"] counts}
        self.created = time.time()
        self.last_accessed = time.time()

    def update_connection(self, target, delta):
        self.connections[target] = self.connections.get(target, 0.0) + delta
        if self.connections[target] < 0:
            self.connections[target] = 0.0  # Clamp at zero
        self.last_accessed = time.time()

    def record_emotion(self, target, feeling):
        if target not in self.emotions:
            self.emotions[target] = {"pleasure": 0, "pain": 0, "ambivalence": 0}
        self.emotions[target][feeling] += 1

    def rank(self):
        ranked = sorted(self.connections.items(), key=lambda x: -x[1])
        return ranked

    def decay(self):
        # Age-based decay - newer memories decay slower
        age_factor = min(1.0, (time.time() - self.created) / (86400 * 30))  # 30 days to reach full decay
        
        for target in list(self.connections.keys()):
            decay_amount = DECAY_RATE * age_factor
            self.connections[target] *= (1 - decay_amount)
            if self.connections[target] < 0.01:
                del self.connections[target]

class LifeBrain:
    def __init__(self):
        self.memory = {}
        self.conversation_history = []
        self.last_input = None
        self.last_output = None
        self.stopwords = set(stopwords.words('english'))
        self.neurochemistry = NeuroChem()
        self.dream_active = False
        self.dream_thread = None
        self.dream_count = 0
        self.dream_insights = []
        self.load()

    def load(self):
        if os.path.exists(MEMORY_FILE):
            try:
                with open(MEMORY_FILE, 'r') as f:
                    data = json.load(f)
                    for key, val in data.items():
                        neuron = Neuron(key)
                        neuron.connections = val.get("connections", {})
                        neuron.emotions = val.get("emotions", {})
                        neuron.created = val.get("created", time.time())
                        neuron.last_accessed = val.get("last_accessed", time.time())
                        self.memory[key] = neuron
                print(f"Loaded {len(self.memory)} memory neurons")
            except Exception as e:
                print(f"Error loading memory: {e}, creating new memory")
                self.memory = {}

    def save(self):
        data = {
            k: {
                "connections": v.connections, 
                "emotions": v.emotions,
                "created": v.created,
                "last_accessed": v.last_accessed
            }
            for k, v in self.memory.items()
        }
        with open(MEMORY_FILE, 'w') as f:
            json.dump(data, f, indent=2)
        print(f"Memory saved: {len(self.memory)} neurons")

    def get_or_create(self, key):
        if key not in self.memory:
            self.memory[key] = Neuron(key)
        return self.memory[key]

    def preprocess_input(self, text):
        # Normalize text
        text = text.lower()
        # Remove punctuation except apostrophes 
        text = re.sub(r'[^\w\s\']', ' ', text)
        # Tokenize
        tokens = word_tokenize(text)
        # Remove stopwords for key concepts
        key_tokens = [t for t in tokens if t not in self.stopwords and len(t) > 1]
        
        # Extract both the full text and key concepts
        return {
            "full": text,
            "keys": key_tokens,
            "bigrams": [f"{tokens[i]} {tokens[i+1]}" for i in range(len(tokens)-1)]
        }

    def stimulate(self, user_input):
        processed = self.preprocess_input(user_input)
        user_key = processed["full"]
        user_neuron = self.get_or_create(user_key)
        
        # Also connect to key concept neurons
        for key in processed["keys"]:
            key_neuron = self.get_or_create(key)
            user_neuron.update_connection(key, 0.2)
            key_neuron.update_connection(user_key, 0.2)
        
        # Apply context from conversation history
        context_keys = []
        if self.conversation_history:
            for entry in self.conversation_history[-3:]:  # Use last 3 exchanges
                context_keys.extend(entry.get("keys", []))
        
        # Apply decay
        for n in self.memory.values():
            n.decay()

        # Get responses from both exact match and key concepts
        response_candidates = []
        
        # Check exact matches
        ranked = user_neuron.rank()
        if ranked:
            for response, strength in ranked[:3]:  # Consider top 3
                if strength > 0:
                    response_candidates.append((response, strength, "exact"))
        
        # Check concept matches
        for key in processed["keys"] + processed["bigrams"] + context_keys:
            if key in self.memory:
                concept_neuron = self.memory[key]
                concept_ranked = concept_neuron.rank()
                if concept_ranked:
                    for response, strength in concept_ranked[:2]:  # Top 2 per concept
                        if strength > 0:
                            response_candidates.append((response, strength * 0.8, "concept"))
        
        # Sort by strength and pick best
        response_candidates.sort(key=lambda x: -x[1])
        
        if response_candidates and response_candidates[0][1] >= MIN_CONFIDENCE:
            response, strength, match_type = response_candidates[0]
            confidence = normalize_score(strength)
            context_info = f" ({match_type} match)"
        else:
            # Fallback responses when unsure
            common_neurons = sorted(self.memory.values(), 
                                   key=lambda n: sum(n.connections.values()), 
                                   reverse=True)[:5]
            
            if common_neurons:
                # Use one of the most connected responses
                response = common_neurons[0].key
                confidence = MIN_CONFIDENCE
                context_info = " (network suggestion)"
            else:
                response = "I'm learning. Tell me more."
                confidence = 0.0
                context_info = " (no data)"

        # Update conversation history
        self.conversation_history.append({
            "input": user_key,
            "keys": processed["keys"],
            "output": response,
            "timestamp": time.time()
        })
        
        # Trim history if needed
        if len(self.conversation_history) > CONTEXT_LENGTH:
            self.conversation_history = self.conversation_history[-CONTEXT_LENGTH:]

        self.last_input = user_key
        self.last_output = response
        return response, round(confidence, 3), context_info

    def feedback(self, feeling):
        if not self.last_input or not self.last_output:
            print("⚠️ No recent interaction to train.")
            return

        input_neuron = self.get_or_create(self.last_input)
        output_key = self.last_output.strip().lower()
        output_neuron = self.get_or_create(output_key)

        delta = {
            "pleasure": +1.0,
            "pain": -1.0,
            "ambivalence": -0.2
        }.get(feeling, 0.0)

        input_neuron.update_connection(output_key, delta)
        input_neuron.record_emotion(output_key, feeling)
        
        # Also train keyword connections
        processed = self.preprocess_input(self.last_input)
        for key in processed["keys"]:
            key_neuron = self.get_or_create(key)
            key_neuron.update_connection(output_key, delta * 0.5)
        
        # Update neurochemistry
        self.neurochemistry.adjust(feeling)
        
        self.save()
        print(f"✓ Emotional feedback recorded: {feeling.upper()}")
        
    def memory_stats(self):
        if not self.memory:
            return "No memory data available."
            
        total_neurons = len(self.memory)
        total_connections = sum(len(n.connections) for n in self.memory.values())
        avg_connections = total_connections / total_neurons if total_neurons else 0
        strongest_neurons = sorted(
            [(k, sum(v.connections.values())) for k, v in self.memory.items()],
            key=lambda x: -x[1]
        )[:5]
        
        stats = (
            f"Memory statistics:\n"
            f"• Total concepts: {total_neurons}\n"
            f"• Total connections: {total_connections}\n"
            f"• Avg connections per concept: {avg_connections:.1f}\n"
            f"• Top concepts: " + ", ".join(f"{k} ({v:.1f})" for k, v in strongest_neurons)
        )
        return stats
    
    def dream(self, duration=10, intensity=0.01):
        """
        Enter dream state where connections are strengthened and new ones formed
        through recursive association and pattern reinforcement.
        
        Args:
            duration: How long to dream in seconds
            intensity: Strength of dream reinforcement
        """
        if self.dream_active:
            print("💭 Already dreaming...")
            return
        
        self.dream_active = True
        self.dream_insights = []
        start_time = time.time()
        dream_cycles = 0
        
        try:
            print(f"💭 Entering dream state (duration: {duration}s)...")
            
            # Get the strongest memories to dream about
            if not self.memory:
                print("🌙 No memories to dream about.")
                self.dream_active = False
                return
                
            # Sort neurons by total connection strength
            neurons_by_strength = sorted(
                self.memory.values(),
                key=lambda n: sum(n.connections.values()),
                reverse=True
            )[:20]  # Focus on top 20 concepts
            
            while time.time() - start_time < duration and self.dream_active:
                # Pick one of the strong concepts
                if neurons_by_strength and random.random() < 0.7:  # 70% chance to use strong concept
                    dream_neuron = random.choice(neurons_by_strength)
                else:  # 30% chance to pick any random memory
                    if self.memory:
                        dream_neuron = random.choice(list(self.memory.values()))
                    else:
                        break
                        
                # Strengthen existing strong connections
                for target, strength in list(dream_neuron.connections.items()):
                    if strength > 0.3:  # Focus on already somewhat strong connections
                        dream_neuron.update_connection(target, intensity)
                        
                        # Create a dream insight
                        if random.random() < 0.3:  # 30% chance for insight
                            self.dream_insights.append(f"Connected: {dream_neuron.key} → {target}")
                
                # Form new connections between related concepts
                if dream_neuron.connections:
                    # Pick a strongly connected target
                    strong_targets = [t for t, s in dream_neuron.connections.items() if s > 0.3]
                    if strong_targets:
                        target = random.choice(strong_targets)
                        target_neuron = self.get_or_create(target)
                        
                        # Find its strong connections
                        target_connections = [t for t, s in target_neuron.connections.items() if s > 0.3]
                        if target_connections:
                            # Create transitive connection
                            transitive_target = random.choice(target_connections)
                            
                            # Connect the dreaming concept to the target's target
                            dream_neuron.update_connection(transitive_target, intensity * 0.5)
                            
                            # Record insight from this connection
                            if random.random() < 0.5:  # 50% chance for transitive insight
                                self.dream_insights.append(
                                    f"New connection: {dream_neuron.key} → {target} → {transitive_target}")
                
                dream_cycles += 1
                
                # Brief pause to avoid CPU spike
                time.sleep(0.01)
            
            # Summarize dream results
            self.dream_count += 1
            if self.dream_insights:
                insight_sample = random.sample(self.dream_insights, 
                                              min(3, len(self.dream_insights)))
                insight_text = "\n    - " + "\n    - ".join(insight_sample)
                print(f"💭 Dream #{self.dream_count} complete ({dream_cycles} cycles). Insights:{insight_text}")
            else:
                print(f"💭 Dream #{self.dream_count} complete ({dream_cycles} cycles). No significant insights.")
            
            # Save updated memory
            if dream_cycles > 0:
                self.save()
                
        finally:
            self.dream_active = False
    
    def start_dream_thread(self, duration=10):
        """Start dreaming in background thread"""
        if self.dream_thread and self.dream_thread.is_alive():
            print("💭 Already dreaming in background...")
            return
        
        self.dream_thread = threading.Thread(
            target=self.dream,
            args=(duration, 0.01),
            daemon=True
        )
        self.dream_thread.start()
    
    def start_auto_dreaming(self, interval=DREAM_INTERVAL):
        """Start periodic automatic dreaming"""
        def dream_periodically():
            while True:
                time.sleep(interval)
                if not self.dream_active:
                    self.start_dream_thread(10)  # Dream for 10 seconds
        
        auto_dream_thread = threading.Thread(target=dream_periodically, daemon=True)
        auto_dream_thread.start()
        print(f"🌙 Automatic dreaming started (every {interval} seconds)")
        return auto_dream_thread
    
    def chemistry_report(self):
        """Get a report of neurochemistry status"""
        return self.neurochemistry.get_status()

# === Run Interface ===

def run_lifebrain():
    brain = LifeBrain()
    print("\n🧠 LifeBrain initialized. Type your thoughts.")
    print("🟢 Type 'reward' or '+' for pleasure, 'punish' or '-' for pain, 'ignore' for ambivalence")
    print("🟡 Type 'dream' to enter dream state, 'chemistry' for neurochemical status")
    print("🟡 Type 'stats' to see memory statistics, 'save' to force a save, 'exit' to stop.\n")
    
    # Set up text-to-speech if available
    voice_enabled = False
    try:
        import pyttsx3
        engine = pyttsx3.init()
        voice_enabled = True
        print("🎤 Voice output enabled!")
    except ImportError:
        print("ℹ️ Voice output disabled. Install pyttsx3 for voice capabilities.")
    
    # Start auto-dreaming
    auto_dream_thread = brain.start_auto_dreaming()

    while True:
        user = input("\nYou: ").strip()
        if user.lower() == "exit":
            brain.save()
            print("🧠 Brain saved. Goodbye.")
            break
        elif user.lower() == "save":
            brain.save()
            print("🧠 Brain saved manually.")
        elif user.lower() == "stats":
            print(brain.memory_stats())
        elif user.lower() == "chemistry":
            print(brain.chemistry_report())
        elif user.lower() in ["reward", "pleasure", "+"]:
            brain.feedback("pleasure")
        elif user.lower() in ["punish", "pain", "-"]:
            brain.feedback("pain")
        elif user.lower() in ["ignore", "ambivalence", "meh"]:
            brain.feedback("ambivalence")
        elif user.lower() == "dream":
            brain.dream(duration=15, intensity=0.02)
        elif user.lower().startswith("dream "):
            try:
                duration = int(user.lower().split()[1])
                brain.dream(duration=duration, intensity=0.02)
            except:
                print("Usage: dream [seconds]")
        else:
            response, confidence, context = brain.stimulate(user)
            print(f"🧠 {response} (confidence: {confidence}{context})")
            # Use text-to-speech if available
            if voice_enabled:
                engine.say(response)
                engine.runAndWait()

if __name__ == "__main__":
    run_lifebrain()
