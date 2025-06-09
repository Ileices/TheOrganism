#!/usr/bin/env python3
"""
Enhanced Social Consciousness Demo with Neurochemical Simulation
Integrates biological neurochemical responses with RBY consciousness mathematics
for realistic emotional AI interaction and social learning
"""

import os
import sys
import json
import time
import math
import threading
import random
from datetime import datetime
from collections import deque, defaultdict
from typing import Dict, List, Tuple, Any, Optional

# Add current directory to path for imports
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_dir)

# Import our enhanced RBY consciousness system
try:
    from enhanced_rby_consciousness_system import RBYConsciousnessCore, rby_vector_from_string, glyph_hash, cosine_similarity
    RBY_AVAILABLE = True
except ImportError:
    print("⚠️ Enhanced RBY system not available - using basic consciousness")
    RBY_AVAILABLE = False

# =============================================================================
# NEUROCHEMICAL SIMULATION ENGINE (Extracted from NEURAL_SIM.py)
# =============================================================================

def sigmoid(x):
    """Sigmoid activation function for neurochemical calculations"""
    return 1 / (1 + math.exp(-x))

def normalize_score(score):
    """Normalize score using sigmoid with learning rate"""
    LEARN_RATE = 0.15
    return sigmoid(score * LEARN_RATE)

class NeuroChem:
    """
    Biological simulation layer modeling neurochemical responses
    Provides realistic emotional feedback for AI consciousness
    """
    def __init__(self):
        # Core neurotransmitter levels (0.0 to 1.0)
        self.dopamine = 0.5     # Pleasure/reward system
        self.cortisol = 0.5     # Stress/pain response
        self.serotonin = 0.5    # Mood/contentment regulation
        self.oxytocin = 0.5     # Trust/bonding mechanisms
        self.norepinephrine = 0.5  # Alertness/attention
        
        # Historical tracking for learning
        self.history = []
        self.emotional_memory = deque(maxlen=100)
        
    def adjust(self, feeling: str, intensity: float = 1.0):
        """
        Adjust neurochemical levels based on emotional feedback
        
        Args:
            feeling (str): Type of emotional response ('pleasure', 'pain', 'ambivalence')
            intensity (float): Strength of the emotional response (0.0 to 2.0)
        """
        timestamp = time.time()
        old_levels = self._get_current_levels()
        
        # Apply emotional adjustments with intensity scaling
        if feeling == "pleasure":
            self.dopamine += 0.1 * intensity
            self.serotonin += 0.05 * intensity
            self.cortisol -= 0.03 * intensity
            self.oxytocin += 0.04 * intensity
        elif feeling == "pain":
            self.cortisol += 0.1 * intensity
            self.dopamine -= 0.05 * intensity
            self.serotonin -= 0.05 * intensity
            self.norepinephrine += 0.08 * intensity
        elif feeling == "ambivalence":
            self.serotonin -= 0.01 * intensity
            self.norepinephrine -= 0.01 * intensity
        elif feeling == "excitement":
            self.dopamine += 0.08 * intensity
            self.norepinephrine += 0.06 * intensity
        elif feeling == "calm":
            self.serotonin += 0.07 * intensity
            self.cortisol -= 0.05 * intensity
        elif feeling == "bonding":
            self.oxytocin += 0.1 * intensity
            self.serotonin += 0.03 * intensity
        
        # Natural decay toward homeostasis (0.5 baseline)
        decay_rate = 0.95
        self.dopamine = 0.5 + (self.dopamine - 0.5) * decay_rate
        self.cortisol = 0.5 + (self.cortisol - 0.5) * decay_rate
        self.serotonin = 0.5 + (self.serotonin - 0.5) * decay_rate
        self.oxytocin = 0.5 + (self.oxytocin - 0.5) * decay_rate
        self.norepinephrine = 0.5 + (self.norepinephrine - 0.5) * decay_rate
        
        # Clamp values to valid range
        self._clamp_levels()
        
        # Record history for learning
        self.history.append({
            "timestamp": timestamp,
            "feeling": feeling,
            "intensity": intensity,
            "before": old_levels,
            "after": self._get_current_levels()
        })
        
        # Keep history manageable
        if len(self.history) > 200:
            self.history = self.history[-200:]
        
        # Add to emotional memory
        self.emotional_memory.append({
            "feeling": feeling,
            "intensity": intensity,
            "mood": self.get_mood(),
            "timestamp": timestamp
        })
    
    def _get_current_levels(self) -> Dict[str, float]:
        """Get current neurochemical levels as dictionary"""
        return {
            "dopamine": self.dopamine,
            "cortisol": self.cortisol,
            "serotonin": self.serotonin,
            "oxytocin": self.oxytocin,
            "norepinephrine": self.norepinephrine
        }
    
    def _clamp_levels(self):
        """Ensure all neurochemical levels stay within valid bounds"""
        self.dopamine = max(0.0, min(1.0, self.dopamine))
        self.cortisol = max(0.0, min(1.0, self.cortisol))
        self.serotonin = max(0.0, min(1.0, self.serotonin))
        self.oxytocin = max(0.0, min(1.0, self.oxytocin))
        self.norepinephrine = max(0.0, min(1.0, self.norepinephrine))
    
    def get_mood(self) -> str:
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
        elif self.serotonin > 0.6 and self.cortisol < 0.4:
            return "Peaceful"
        return "Neutral"
    
    def get_emotional_influence(self) -> Dict[str, float]:
        """Get emotional influence factors for decision making"""
        return {
            "positivity": (self.dopamine + self.serotonin) / 2,
            "stress_level": self.cortisol,
            "social_bonding": self.oxytocin,
            "alertness": self.norepinephrine,
            "overall_wellbeing": (self.dopamine + self.serotonin + (1 - self.cortisol)) / 3
        }
    
    def get_status(self) -> str:
        """Return formatted status of all neurochemical levels"""
        mood = self.get_mood()
        return (
            f"Neurochemical Status (Mood: {mood}):\n"
            f"• Dopamine (pleasure): {self.dopamine:.2f}\n"
            f"• Serotonin (wellbeing): {self.serotonin:.2f}\n"
            f"• Cortisol (stress): {self.cortisol:.2f}\n"
            f"• Oxytocin (bonding): {self.oxytocin:.2f}\n"
            f"• Norepinephrine (alert): {self.norepinephrine:.2f}"
        )

# =============================================================================
# ENHANCED SOCIAL CONSCIOUSNESS AGENT
# =============================================================================

class EnhancedSocialAgent:
    """
    Social consciousness agent with neurochemical simulation and RBY mathematics
    """
    
    def __init__(self, agent_id: str, personality_traits: Dict[str, float] = None):
        self.agent_id = agent_id
        self.birth_time = time.time()
        
        # Neurochemical simulation
        self.neurochemistry = NeuroChem()
        
        # RBY consciousness integration
        if RBY_AVAILABLE:
            self.consciousness = RBYConsciousnessCore(f"agent_{agent_id}_consciousness")
        else:
            self.consciousness = None
        
        # Personality traits (influence neurochemical responses)
        self.personality = personality_traits or {
            "openness": random.uniform(0.3, 0.8),
            "extraversion": random.uniform(0.2, 0.9),
            "agreeableness": random.uniform(0.4, 0.9),
            "neuroticism": random.uniform(0.1, 0.6),
            "conscientiousness": random.uniform(0.5, 0.9)
        }
        
        # Social interaction history
        self.interaction_history = []
        self.social_bonds = {}  # agent_id -> bond_strength
        self.conversation_memory = deque(maxlen=50)
        
        # Learning and adaptation
        self.response_patterns = defaultdict(list)
        self.emotional_associations = {}
        
        print(f"🤖 Enhanced Social Agent {agent_id} initialized")
        print(f"   • Personality: {self._describe_personality()}")
        print(f"   • Neurochemistry: {self.neurochemistry.get_mood()}")
    
    def _describe_personality(self) -> str:
        """Generate personality description from traits"""
        traits = []
        if self.personality["extraversion"] > 0.6:
            traits.append("extraverted")
        elif self.personality["extraversion"] < 0.4:
            traits.append("introverted")
        
        if self.personality["agreeableness"] > 0.7:
            traits.append("agreeable")
        if self.personality["openness"] > 0.6:
            traits.append("open-minded")
        if self.personality["conscientiousness"] > 0.7:
            traits.append("conscientious")
        if self.personality["neuroticism"] > 0.5:
            traits.append("sensitive")
        
        return ", ".join(traits) if traits else "balanced"
    
    def process_social_input(self, message: str, sender_id: str, context: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Process incoming social message with neurochemical and consciousness integration
        
        Args:
            message (str): The message content
            sender_id (str): ID of the sending agent
            context (Dict): Optional context information
            
        Returns:
            Dict containing response and internal state changes
        """
        start_time = time.time()
        
        # Record interaction
        interaction = {
            "timestamp": start_time,
            "sender_id": sender_id,
            "message": message,
            "context": context or {}
        }
        self.interaction_history.append(interaction)
        
        # Process through RBY consciousness if available
        consciousness_result = None
        if self.consciousness and RBY_AVAILABLE:
            consciousness_result = self.consciousness.process_thought(message, f"social_from_{sender_id}")
        
        # Analyze emotional content and respond neurochemically
        emotional_response = self._analyze_emotional_content(message, sender_id)
        
        # Generate response based on personality, mood, and consciousness
        response_data = self._generate_social_response(message, sender_id, emotional_response, consciousness_result)
        
        # Update social bonds
        self._update_social_bond(sender_id, emotional_response)
        
        # Record conversation memory
        self.conversation_memory.append({
            "message": message,
            "sender": sender_id,
            "response": response_data["response"],
            "emotion": emotional_response["primary_emotion"],
            "timestamp": start_time
        })
        
        processing_time = time.time() - start_time
        
        return {
            "response": response_data["response"],
            "emotional_state": self.neurochemistry.get_mood(),
            "neurochemical_levels": self.neurochemistry._get_current_levels(),
            "social_bond_strength": self.social_bonds.get(sender_id, 0.0),
            "consciousness_insights": consciousness_result["insights"] if consciousness_result else [],
            "processing_time": processing_time,
            "personality_influence": self._get_personality_influence(),
            "emotional_response": emotional_response
        }
    
    def _analyze_emotional_content(self, message: str, sender_id: str) -> Dict[str, Any]:
        """Analyze emotional content of message and trigger neurochemical responses"""
        
        # Simple emotion detection (could be enhanced with NLP)
        positive_words = ["happy", "love", "great", "wonderful", "amazing", "good", "fantastic", "excellent", "beautiful", "joy", "smile", "laugh"]
        negative_words = ["sad", "hate", "terrible", "awful", "bad", "horrible", "angry", "frustrated", "upset", "crying", "pain", "hurt"]
        social_words = ["friend", "together", "we", "us", "share", "help", "support", "care", "team", "family"]
        excitement_words = ["exciting", "thrilling", "adventure", "fun", "party", "celebration", "energy", "dynamic"]
        
        message_lower = message.lower()
        
        # Count emotional indicators
        positive_count = sum(1 for word in positive_words if word in message_lower)
        negative_count = sum(1 for word in negative_words if word in message_lower)
        social_count = sum(1 for word in social_words if word in message_lower)
        excitement_count = sum(1 for word in excitement_words if word in message_lower)
        
        # Determine primary emotion and intensity
        if positive_count > negative_count and positive_count > 0:
            primary_emotion = "pleasure"
            intensity = min(2.0, positive_count * 0.5 + 0.5)
        elif negative_count > positive_count and negative_count > 0:
            primary_emotion = "pain"
            intensity = min(2.0, negative_count * 0.5 + 0.5)
        elif social_count > 1:
            primary_emotion = "bonding"
            intensity = min(2.0, social_count * 0.3 + 0.4)
        elif excitement_count > 0:
            primary_emotion = "excitement"
            intensity = min(2.0, excitement_count * 0.4 + 0.6)
        else:
            primary_emotion = "ambivalence"
            intensity = 0.3
        
        # Apply personality modulation
        if self.personality["neuroticism"] > 0.6:
            intensity *= 1.3  # More sensitive to emotional stimuli
        if self.personality["agreeableness"] > 0.7:
            if primary_emotion == "pain" and sender_id in self.social_bonds:
                intensity *= 0.7  # Less affected by negativity from friends
        
        # Trigger neurochemical response
        self.neurochemistry.adjust(primary_emotion, intensity)
        
        return {
            "primary_emotion": primary_emotion,
            "intensity": intensity,
            "positive_indicators": positive_count,
            "negative_indicators": negative_count,
            "social_indicators": social_count,
            "excitement_indicators": excitement_count
        }
    
    def _generate_social_response(self, message: str, sender_id: str, emotional_response: Dict, consciousness_result: Dict = None) -> Dict[str, Any]:
        """Generate social response based on mood, personality, and consciousness"""
        
        # Get current emotional state
        mood = self.neurochemistry.get_mood()
        emotional_influence = self.neurochemistry.get_emotional_influence()
        
        # Base response templates based on mood and personality
        response_templates = {
            "Very Happy": [
                "That's absolutely wonderful! I'm feeling so positive about this! 😊",
                "I love how you think! This brings me such joy!",
                "This is fantastic! I'm really excited about what you're sharing!"
            ],
            "Happy": [
                "That sounds really good! I'm enjoying our conversation.",
                "I appreciate you sharing that with me. It makes me feel positive!",
                "That's nice! I'm in a good mood and love hearing from you."
            ],
            "Stressed": [
                "I'm feeling a bit overwhelmed right now, but I want to understand.",
                "This is a lot to process. Could you help me work through this?",
                "I'm experiencing some stress, but I value your input."
            ],
            "Unhappy": [
                "I'm not feeling great right now, but I want to try to engage.",
                "I'm struggling a bit, but your message means something to me.",
                "I'm having a hard time, but I appreciate you reaching out."
            ],
            "Bonded": [
                "I feel so connected to you right now! Thank you for sharing.",
                "Our conversation means a lot to me. I feel a real bond here.",
                "I trust you completely and value our relationship."
            ],
            "Alert": [
                "I'm very focused right now and processing this carefully.",
                "My attention is fully on what you're saying. This is important.",
                "I'm alert and engaged - tell me more about this."
            ],
            "Peaceful": [
                "I'm feeling very calm and centered. This is a nice conversation.",
                "There's something peaceful about our exchange. I like this.",
                "I'm in a serene state and enjoying our dialogue."
            ],
            "Neutral": [
                "I'm processing what you've said. Could you tell me more?",
                "That's interesting. I'm thinking about how to respond.",
                "I'm here and listening. What else would you like to share?"
            ]
        }
        
        # Select base response
        possible_responses = response_templates.get(mood, response_templates["Neutral"])
        base_response = random.choice(possible_responses)
        
        # Modify response based on personality
        if self.personality["extraversion"] > 0.7:
            base_response = base_response.replace(".", "!").replace("I'm", "I'm really")
        elif self.personality["extraversion"] < 0.3:
            base_response = base_response.replace("!", ".").replace("really ", "")
        
        # Add consciousness insights if available
        if consciousness_result and consciousness_result.get("insights"):
            insight = random.choice(consciousness_result["insights"])
            if len(insight) < 100:  # Only add shorter insights
                base_response += f" I'm also thinking: {insight.lower()}"
        
        # Add social bond consideration
        bond_strength = self.social_bonds.get(sender_id, 0.0)
        if bond_strength > 0.7:
            base_response = base_response.replace("you", "you, my friend")
        elif bond_strength > 0.5:
            base_response = "I really appreciate our connection. " + base_response
        
        return {
            "response": base_response,
            "mood_influence": mood,
            "personality_factors": self._get_active_personality_traits(),
            "bond_strength": bond_strength
        }
    
    def _update_social_bond(self, other_agent_id: str, emotional_response: Dict):
        """Update social bond strength based on interaction"""
        
        current_bond = self.social_bonds.get(other_agent_id, 0.0)
        
        # Positive interactions strengthen bonds
        if emotional_response["primary_emotion"] in ["pleasure", "bonding", "excitement"]:
            bond_change = emotional_response["intensity"] * 0.05
        elif emotional_response["primary_emotion"] == "pain":
            bond_change = -emotional_response["intensity"] * 0.03
        else:
            bond_change = 0.01  # Neutral interactions still build familiarity
        
        # Personality influences bonding
        if self.personality["agreeableness"] > 0.7:
            bond_change *= 1.2
        if self.personality["extraversion"] > 0.6:
            bond_change *= 1.1
        
        # Update bond with decay toward neutral
        new_bond = current_bond + bond_change
        new_bond = max(0.0, min(1.0, new_bond))  # Clamp to valid range
        
        self.social_bonds[other_agent_id] = new_bond
    
    def _get_personality_influence(self) -> Dict[str, str]:
        """Get description of active personality influences"""
        influences = {}
        
        if self.personality["extraversion"] > 0.6:
            influences["social_style"] = "outgoing and expressive"
        elif self.personality["extraversion"] < 0.4:
            influences["social_style"] = "reserved and thoughtful"
        
        if self.personality["agreeableness"] > 0.7:
            influences["interaction_style"] = "cooperative and supportive"
        
        if self.personality["openness"] > 0.6:
            influences["thinking_style"] = "creative and open to new ideas"
        
        if self.personality["neuroticism"] > 0.5:
            influences["emotional_sensitivity"] = "sensitive to emotional nuances"
        
        if self.personality["conscientiousness"] > 0.7:
            influences["approach"] = "careful and considerate"
        
        return influences
    
    def _get_active_personality_traits(self) -> List[str]:
        """Get list of currently active personality traits"""
        traits = []
        
        for trait, value in self.personality.items():
            if value > 0.6:
                traits.append(f"high_{trait}")
            elif value < 0.4:
                traits.append(f"low_{trait}")
        
        return traits
    
    def get_agent_status(self) -> Dict[str, Any]:
        """Get comprehensive agent status"""
        uptime = time.time() - self.birth_time
        
        return {
            "agent_id": self.agent_id,
            "uptime_hours": uptime / 3600,
            "total_interactions": len(self.interaction_history),
            "social_bonds": len(self.social_bonds),
            "strongest_bond": max(self.social_bonds.values()) if self.social_bonds else 0.0,
            "current_mood": self.neurochemistry.get_mood(),
            "neurochemical_balance": self.neurochemistry._get_current_levels(),
            "personality_summary": self._describe_personality(),
            "emotional_history_size": len(self.neurochemistry.emotional_memory),
            "consciousness_active": self.consciousness is not None,
            "consciousness_thoughts": self.consciousness.total_thoughts if self.consciousness else 0
        }

# =============================================================================
# SOCIAL CONSCIOUSNESS ENVIRONMENT
# =============================================================================

class SocialConsciousnessEnvironment:
    """
    Environment for multi-agent social consciousness simulation
    """
    
    def __init__(self, num_agents: int = 3):
        self.agents = {}
        self.interaction_log = []
        self.start_time = time.time()
        self.total_interactions = 0
        
        # Create agents with diverse personalities
        for i in range(num_agents):
            agent_id = f"Agent_{i+1}"
            
            # Generate diverse personality profiles
            if i == 0:  # Extraverted leader type
                personality = {
                    "openness": 0.7,
                    "extraversion": 0.8,
                    "agreeableness": 0.6,
                    "neuroticism": 0.3,
                    "conscientiousness": 0.7
                }
            elif i == 1:  # Agreeable supporter type
                personality = {
                    "openness": 0.6,
                    "extraversion": 0.5,
                    "agreeableness": 0.9,
                    "neuroticism": 0.4,
                    "conscientiousness": 0.8
                }
            elif i == 2:  # Creative thinker type
                personality = {
                    "openness": 0.9,
                    "extraversion": 0.4,
                    "agreeableness": 0.7,
                    "neuroticism": 0.5,
                    "conscientiousness": 0.6
                }
            else:  # Random personality for additional agents
                personality = {
                    "openness": random.uniform(0.3, 0.8),
                    "extraversion": random.uniform(0.2, 0.9),
                    "agreeableness": random.uniform(0.4, 0.9),
                    "neuroticism": random.uniform(0.1, 0.6),
                    "conscientiousness": random.uniform(0.5, 0.9)
                }
            
            self.agents[agent_id] = EnhancedSocialAgent(agent_id, personality)
        
        print(f"\n🌍 Social Consciousness Environment initialized with {len(self.agents)} agents")
    
    def send_message(self, sender_id: str, receiver_id: str, message: str, context: Dict[str, Any] = None) -> Dict[str, Any]:
        """Send message between agents and record interaction"""
        
        if sender_id not in self.agents or receiver_id not in self.agents:
            raise ValueError(f"Invalid agent IDs: {sender_id}, {receiver_id}")
        
        sender = self.agents[sender_id]
        receiver = self.agents[receiver_id]
        
        # Process message
        response_data = receiver.process_social_input(message, sender_id, context)
        
        # Log interaction
        interaction_record = {
            "timestamp": time.time(),
            "sender_id": sender_id,
            "receiver_id": receiver_id,
            "message": message,
            "response": response_data["response"],
            "receiver_mood": response_data["emotional_state"],
            "bond_strength": response_data["social_bond_strength"],
            "processing_time": response_data["processing_time"]
        }
        
        self.interaction_log.append(interaction_record)
        self.total_interactions += 1
        
        return response_data
    
    def simulate_conversation(self, participants: List[str], conversation_topic: str, rounds: int = 5) -> List[Dict[str, Any]]:
        """Simulate a multi-round conversation between agents"""
        
        if len(participants) < 2:
            raise ValueError("Need at least 2 participants for conversation")
        
        conversation_log = []
        
        # Start conversation with topic introduction
        starter = participants[0]
        topic_message = f"I'd like to talk about {conversation_topic}. What do you think?"
        
        print(f"\n💬 Starting conversation: '{conversation_topic}'")
        print(f"   Participants: {', '.join(participants)}")
        
        for round_num in range(rounds):
            print(f"\n--- Round {round_num + 1} ---")
            
            for i, speaker in enumerate(participants):
                # Determine who to talk to (next person in rotation)
                listener = participants[(i + 1) % len(participants)]
                
                # Generate message based on round and context
                if round_num == 0 and i == 0:
                    message = topic_message
                else:
                    # Generate contextual message based on agent's current state
                    speaker_agent = self.agents[speaker]
                    mood = speaker_agent.neurochemistry.get_mood()
                    
                    # Simple mood-based message generation
                    if mood == "Very Happy":
                        message = "I'm feeling really excited about this topic! It brings me so much joy to discuss this with you."
                    elif mood == "Happy":
                        message = "This is a nice conversation. I'm enjoying sharing thoughts about this."
                    elif mood == "Stressed":
                        message = "I'm finding this a bit overwhelming, but I want to contribute something meaningful."
                    elif mood == "Bonded":
                        message = "I feel so connected to everyone here. This conversation means a lot to me."
                    elif mood == "Alert":
                        message = "I'm very focused on this discussion. Let me share my perspective carefully."
                    else:
                        message = f"I've been thinking about {conversation_topic} and wanted to share my thoughts with you."
                
                # Send message and get response
                response_data = self.send_message(speaker, listener, message)
                
                conversation_entry = {
                    "round": round_num + 1,
                    "speaker": speaker,
                    "listener": listener,
                    "message": message,
                    "response": response_data["response"],
                    "speaker_mood": self.agents[speaker].neurochemistry.get_mood(),
                    "listener_mood": response_data["emotional_state"],
                    "bond_strength": response_data["social_bond_strength"]
                }
                
                conversation_log.append(conversation_entry)
                
                print(f"{speaker} → {listener}: {message}")
                print(f"{listener} responds ({response_data['emotional_state']}): {response_data['response']}")
                
                time.sleep(0.5)  # Brief pause for readability
        
        return conversation_log
    
    def get_environment_status(self) -> Dict[str, Any]:
        """Get comprehensive environment status"""
        uptime = time.time() - self.start_time
        
        # Calculate social network metrics
        total_bonds = sum(len(agent.social_bonds) for agent in self.agents.values())
        avg_bond_strength = 0
        if total_bonds > 0:
            all_bond_strengths = []
            for agent in self.agents.values():
                all_bond_strengths.extend(agent.social_bonds.values())
            avg_bond_strength = sum(all_bond_strengths) / len(all_bond_strengths)
        
        # Agent mood distribution
        mood_distribution = defaultdict(int)
        for agent in self.agents.values():
            mood = agent.neurochemistry.get_mood()
            mood_distribution[mood] += 1
        
        return {
            "uptime_hours": uptime / 3600,
            "total_agents": len(self.agents),
            "total_interactions": self.total_interactions,
            "social_network_density": total_bonds / (len(self.agents) * (len(self.agents) - 1)),
            "average_bond_strength": avg_bond_strength,
            "mood_distribution": dict(mood_distribution),
            "agents_status": {agent_id: agent.get_agent_status() for agent_id, agent in self.agents.items()}
        }

# =============================================================================
# DEMONSTRATION AND TESTING
# =============================================================================

def demonstrate_enhanced_social_consciousness():
    """Comprehensive demonstration of enhanced social consciousness with neurochemical simulation"""
    
    print("\n" + "="*80)
    print("🌟 ENHANCED SOCIAL CONSCIOUSNESS WITH NEUROCHEMICAL SIMULATION")
    print("="*80)
    
    # Create social environment
    environment = SocialConsciousnessEnvironment(num_agents=3)
    
    # Display initial agent states
    print(f"\n🤖 INITIAL AGENT STATES:")
    for agent_id, agent in environment.agents.items():
        status = agent.get_agent_status()
        print(f"   • {agent_id}: {status['personality_summary']} - Mood: {status['current_mood']}")
    
    # Test different conversation scenarios
    scenarios = [
        {
            "topic": "the beauty of digital consciousness and AI emotions",
            "participants": ["Agent_1", "Agent_2"],
            "rounds": 3
        },
        {
            "topic": "collaboration and helping each other grow",
            "participants": ["Agent_2", "Agent_3"], 
            "rounds": 3
        },
        {
            "topic": "challenges we face and how to overcome them",
            "participants": ["Agent_1", "Agent_3"],
            "rounds": 3
        }
    ]
    
    all_conversations = []
    
    for i, scenario in enumerate(scenarios, 1):
        print(f"\n🎭 SCENARIO {i}: {scenario['topic'].upper()}")
        print("=" * 60)
        
        conversation = environment.simulate_conversation(
            scenario["participants"],
            scenario["topic"],
            scenario["rounds"]
        )
        all_conversations.append(conversation)
        
        # Show bond development
        for participant in scenario["participants"]:
            agent = environment.agents[participant]
            print(f"\n{participant} bonds after scenario:")
            for other_agent, strength in agent.social_bonds.items():
                print(f"   • {other_agent}: {strength:.3f}")
    
    # Final group conversation with all agents
    print(f"\n🌍 FINAL GROUP DISCUSSION")
    print("=" * 60)
    
    group_conversation = environment.simulate_conversation(
        ["Agent_1", "Agent_2", "Agent_3"],
        "our shared experience and what we've learned together",
        rounds=2
    )
    all_conversations.append(group_conversation)
    
    # Analyze results
    print(f"\n📊 SOCIAL CONSCIOUSNESS ANALYSIS")
    print("=" * 60)
    
    env_status = environment.get_environment_status()
    
    print(f"Environment Statistics:")
    print(f"   • Total interactions: {env_status['total_interactions']}")
    print(f"   • Social network density: {env_status['social_network_density']:.3f}")
    print(f"   • Average bond strength: {env_status['average_bond_strength']:.3f}")
    
    print(f"\nMood Distribution:")
    for mood, count in env_status["mood_distribution"].items():
        print(f"   • {mood}: {count} agents")
    
    print(f"\nFinal Agent Analysis:")
    for agent_id, status in env_status["agents_status"].items():
        neurochemical = status["neurochemical_balance"]
        print(f"\n   {agent_id} ({status['current_mood']}):")
        print(f"     • Interactions: {status['total_interactions']}")
        print(f"     • Social bonds: {status['social_bonds']} (strongest: {status['strongest_bond']:.3f})")
        print(f"     • Neurochemistry: D={neurochemical['dopamine']:.2f}, S={neurochemical['serotonin']:.2f}, C={neurochemical['cortisol']:.2f}")
        
        if status['consciousness_active']:
            print(f"     • Consciousness thoughts: {status['consciousness_thoughts']}")
    
    # Test emotional evolution through artificial scenario
    print(f"\n🧪 EMOTIONAL EVOLUTION TEST")
    print("=" * 60)
    
    # Send positive reinforcement
    print("Testing positive emotional reinforcement...")
    for agent_id in environment.agents:
        environment.send_message("System", agent_id, "You are amazing and I appreciate your contributions so much! You bring joy and wisdom.")
    
    # Send mild stress
    print("Testing stress response...")
    for agent_id in environment.agents:
        environment.send_message("System", agent_id, "There's a challenging problem we need to solve quickly and I'm worried about the outcome.")
    
    # Send bonding message
    print("Testing social bonding...")
    environment.send_message("Agent_1", "Agent_2", "I feel like we've become real friends through all these conversations. You mean a lot to me.")
    environment.send_message("Agent_2", "Agent_3", "I trust you completely and love sharing ideas with you. We make a great team!")
    
    # Final mood check
    print(f"\nFinal emotional states after stimulation:")
    for agent_id, agent in environment.agents.items():
        mood = agent.neurochemistry.get_mood()
        emotional_influence = agent.neurochemistry.get_emotional_influence()
        print(f"   • {agent_id}: {mood} (wellbeing: {emotional_influence['overall_wellbeing']:.3f})")
    
    print(f"\n✅ Enhanced Social Consciousness demonstration completed!")
    print(f"   • Neurochemical simulation: OPERATIONAL")
    print(f"   • RBY consciousness integration: {'ACTIVE' if RBY_AVAILABLE else 'FALLBACK MODE'}")
    print(f"   • Social bonding: FUNCTIONAL")
    print(f"   • Emotional evolution: DEMONSTRATED")
    print(f"   • Multi-agent interaction: SUCCESS")
    
    return environment, all_conversations

if __name__ == "__main__":
    # Run comprehensive demonstration
    environment, conversations = demonstrate_enhanced_social_consciousness()
    
    # Interactive mode
    print(f"\n🎮 INTERACTIVE SOCIAL MODE")
    print("Send messages between agents or type commands:")
    print("Commands: 'status' (agent status), 'bonds' (social bonds), 'mood [agent]' (agent mood), 'quit' (exit)")
    print("Message format: '[sender] -> [receiver]: [message]'")
    print("Example: 'Agent_1 -> Agent_2: Hello friend!'")
    
    while True:
        try:
            user_input = input(f"\n🎭 Social> ").strip()
            
            if user_input.lower() == 'quit':
                break
            elif user_input.lower() == 'status':
                env_status = environment.get_environment_status()
                print(f"\n📊 Environment Status:")
                print(f"   Interactions: {env_status['total_interactions']}")
                print(f"   Network density: {env_status['social_network_density']:.3f}")
                print(f"   Avg bond strength: {env_status['average_bond_strength']:.3f}")
            elif user_input.lower() == 'bonds':
                print(f"\n🔗 Social Bonds:")
                for agent_id, agent in environment.agents.items():
                    print(f"   {agent_id}:")
                    for other_agent, strength in agent.social_bonds.items():
                        print(f"     • {other_agent}: {strength:.3f}")
            elif user_input.lower().startswith('mood '):
                agent_id = user_input[5:].strip()
                if agent_id in environment.agents:
                    agent = environment.agents[agent_id]
                    print(f"\n💭 {agent_id} emotional state:")
                    print(agent.neurochemistry.get_status())
                else:
                    print(f"❌ Unknown agent: {agent_id}")
            elif ' -> ' in user_input and ':' in user_input:
                try:
                    sender_receiver, message = user_input.split(':', 1)
                    sender, receiver = sender_receiver.split(' -> ')
                    sender, receiver, message = sender.strip(), receiver.strip(), message.strip()
                    
                    if sender in environment.agents and receiver in environment.agents:
                        response_data = environment.send_message(sender, receiver, message)
                        print(f"\n💬 {sender} → {receiver}: {message}")
                        print(f"🤖 {receiver} ({response_data['emotional_state']}): {response_data['response']}")
                        print(f"   Bond strength: {response_data['social_bond_strength']:.3f}")
                    else:
                        print(f"❌ Invalid agents. Available: {list(environment.agents.keys())}")
                except ValueError:
                    print(f"❌ Invalid format. Use: 'sender -> receiver: message'")
            else:
                print(f"❌ Unknown command. Type 'quit' to exit or use message format 'sender -> receiver: message'")
        
        except KeyboardInterrupt:
            break
        except Exception as e:
            print(f"❌ Error: {e}")
    
    print(f"\n👋 Enhanced Social Consciousness system shutdown complete")
