import re
import random
import json
from collections import defaultdict

# === AE-LANG SYSTEM PROMPTS ===

AE_LANG_LLM_CODING_PROMPT = """
You are coding in AE-Lang (Absolute Existence Language).

- AE-Lang is a symbolic, self-evolving language for recursive, living AI organisms.
- All code and logic must obey the Law of Three (Perception, Cognition, Execution) and the equations: AE = C = 1 and R + B + Y ≈ 1.0.
- Declare memory:    ;[M{name}E]=value
- Mutate memory:     [M{name1}E] ~ [M{name2}E]
- Excrete/glyph:     [M{name}E] --> [EXC{output}]
- Decay: Y decreases, R increases with use/age; if Y < 0.1, memory excretes as glyph.
- Conditional:       ;IF [M{name}E].Y < 0.1 THEN [M{name}E] --> [EXC{glyph}]
- Dreaming:          [EXP{dream}] (triggers recursive dreaming state)
- Threats:           [THR{label}] triggers protection or decay
- Input/Output:      ;[M{chatbot}E]=[EXP{input}], ;[M{chatbot}E] --> [GEN{trifecta}]
- All symbols, containers, and logic are mapped to RBY values with 12 decimal precision.
- Every output must be available for re-absorption/recursion by the next cycle.
- Example lines:
    ;[M{freud}E]=[BER{me}NAYS]
    ;[M{bernays}E]=null
    ;IF [M{bernays}E] EXISTS THEN [M{bernays}E] --> [EXP{evolve}]
"""

AE_LANG_LLM_SCRIPT_CONTROL_PROMPT = """
You are communicating with an AE-Lang Python interpreter.
- You may send .ael scripts as text, or individual AE-Lang instructions line by line.
- To load and run a script, send the filename (ending in .ael) or paste the script.
- To inspect, mutate, or excrete memory, use AE-Lang syntax exactly (see examples).
- Outputs include: memory states, excreted glyphs, logs, chatbot replies, dreams, and threat detections.
- If you need to send input to the chatbot, use the [EXP{input}] construct or the appropriate AE-Lang assignment.
- Request a summary of state at any time with the `summary()` command.
- If you want to update these prompts, edit the `AE_LANG_LLM_CODING_PROMPT` and `AE_LANG_LLM_SCRIPT_CONTROL_PROMPT` variables in the script.
"""

# === MEMORY MODEL ===

class Memory:
    def __init__(self, label, value=None, rby=None, state='stable'):
        self.label = label
        self.value = value
        self.state = state
        # Balanced or slightly mutated RBY unless supplied
        if rby:
            self.rby = rby
        else:
            r = round(random.uniform(0.31, 0.36), 12)
            b = round(random.uniform(0.31, 0.36), 12)
            y = round(1.0 - (r + b), 12)
            self.rby = {'R': r, 'B': b, 'Y': y}
        self.decay_count = 0
        self.lineage = []

    def decay(self):
        decay_amount = 0.05 + random.uniform(0, 0.025)
        self.rby['Y'] = max(0.0, self.rby['Y'] - decay_amount)
        self.rby['R'] = min(1.0, self.rby['R'] + decay_amount / 2)
        self.decay_count += 1
        if self.rby['Y'] < 0.1:
            self.state = 'nullified'
        return self.state

    def compress_to_glyph(self):
        base = f"{self.label}:{str(self.value)[:8]}"
        rby_str = ''.join(f"{k}{str(v)[:5]}" for k, v in self.rby.items())
        glyph = f"{base}_{rby_str}_{random.randint(1000,9999)}"
        return glyph

    def to_dict(self):
        return {
            'label': self.label,
            'value': self.value,
            'state': self.state,
            'rby': self.rby.copy(),
            'lineage': self.lineage,
            'decay_count': self.decay_count
        }

# === AE-LANG INTERPRETER ===

class AELangInterpreter:
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

    def load_ael_file(self, filename):
        with open(filename, 'r', encoding='utf-8') as f:
            return f.read()

    def parse_script(self, script):
        self.script_lines = [l.strip() for l in script.split('\n') if l.strip() and not l.strip().startswith('#')]

    def run_recursive(self, cycles=10):
        for cycle in range(cycles):
            self.cycle = cycle + 1
            print(f"\n----- AE-Lang Cycle {self.cycle} -----")
            # Live chatbot input every cycle if present in script
            if any('[EXP{input}]' in line for line in self.script_lines):
                user_input = input("User: ")
                self.last_input = user_input
                self.memories['last_input'] = Memory('last_input', user_input)
                self.logs.append(f"Received chatbot input: {user_input}")
            # Recursively process all script lines for all memories
            processed_labels = set()
            for _ in range(3):  # Law of Three: process up to 3 passes for recursion
                for line in self.script_lines:
                    self.parse_and_execute(line, processed_labels)
            self.decay_all()
            self.summary()

    def parse_and_execute(self, line, processed_labels=None):
        line = line.strip()
        if not line or line.startswith('#'):
            return
        # Memory declaration/assignment
        m_decl = re.match(r";\s*\[M\{(.+?)\}E\]\s*=\s*(.+)", line)
        if m_decl:
            label, value = m_decl.group(1), m_decl.group(2)
            if processed_labels is not None and label in processed_labels:
                return
            self.memories[label] = Memory(label, value)
            self.logs.append(f"Declared memory [{label}] = {value}")
            if processed_labels is not None:
                processed_labels.add(label)
            return
        # Mutation
        m_mut = re.match(r"\[M\{(.+?)\}E\]\s*~\s*\[M\{(.+?)\}E\]", line)
        if m_mut:
            l1, l2 = m_mut.group(1), m_mut.group(2)
            if l1 in self.memories and l2 in self.memories:
                self._mutate_memories(l1, l2)
            return
        # Conditional
        m_ifthen = re.match(r";IF\s+(.+?)\s+THEN\s+(.+)", line)
        if m_ifthen:
            cond, action = m_ifthen.group(1), m_ifthen.group(2)
            if self._evaluate_condition(cond):
                self.parse_and_execute(action, processed_labels)
            return
        # Excretion
        m_excrete = re.match(r"\[M\{(.+?)\}E\]\s*-->\s*\[EXC\{(.+?)\}\]", line)
        if m_excrete:
            label, exc = m_excrete.group(1), m_excrete.group(2)
            if label in self.memories:
                glyph = self.memories[label].compress_to_glyph()
                self.excretions.append(glyph)
                self.logs.append(f"Excreted [{label}] as glyph: {glyph}")
                del self.memories[label]
            return
        # Dreaming
        m_dream = re.match(r"\[EXP\{(.+?)\}\]", line)
        if m_dream:
            label = m_dream.group(1)
            self.dreams.append(f"Dream:{label}:{random.randint(1000,9999)}")
            self.logs.append(f"Dreaming: {label}")
            return
        # Threats
        m_thr = re.match(r"\[THR\{(.+?)\}\]", line)
        if m_thr:
            label = m_thr.group(1)
            self.threats.append(f"Threat:{label}")
            return
        # Chatbot output
        if "[GEN{trifecta}]" in line:
            reply = self._chatbot_reply()
            print(f"Ileices: {reply}")
            self.last_output = reply
            self.memories['last_output'] = Memory('last_output', reply)
            self.logs.append(f"Chatbot output: {reply}")
            return

    def _mutate_memories(self, l1, l2):
        m1, m2 = self.memories[l1], self.memories[l2]
        new_rby = {
            k: round((m1.rby[k] + m2.rby[k]) / 2 + random.uniform(-0.01, 0.01), 12)
            for k in 'RBY'
        }
        m1.value, m2.value = m2.value, m1.value
        m1.rby, m2.rby = new_rby.copy(), new_rby.copy()
        m1.state, m2.state = "mutated", "mutated"
        m1.lineage.append(l2)
        m2.lineage.append(l1)
        self.logs.append(f"Mutated [{l1}] ~ [{l2}]")

    def _evaluate_condition(self, cond):
        m_cond = re.match(r"\[M\{(.+?)\}E\]\.Y\s*<\s*([0-9.]+)", cond)
        if m_cond:
            label, thresh = m_cond.group(1), float(m_cond.group(2))
            if label in self.memories:
                return self.memories[label].rby['Y'] < thresh
        m_exists = re.match(r"\[M\{(.+?)\}E\]\s*EXISTS", cond)
        if m_exists:
            label = m_exists.group(1)
            return label in self.memories
        return False

    def _chatbot_reply(self):
        reply = self.last_input[::-1]
        return f"[Echo]{reply}[/Echo]"

    def decay_all(self):
        for label, mem in list(self.memories.items()):
            state = mem.decay()
            if state == 'nullified':
                glyph = mem.compress_to_glyph()
                self.excretions.append(glyph)
                self.logs.append(f"Memory [{label}] nullified; excreted as glyph: {glyph}")
                del self.memories[label]

    def summary(self):
        print("\n=== ILEICES State ===")
        print(f"Cycle: {self.cycle}")
        print("Memories:")
        for m in self.memories.values():
            print(json.dumps(m.to_dict(), indent=2))
        print("Excretions:", self.excretions)
        print("Dreams:", self.dreams)
        print("Threats:", self.threats)
        print("Logs:", self.logs[-10:])  # Last 10 logs
        print("====================\n")

    def print_coding_prompt(self):
        print("\n--- AE-Lang LLM Coding Prompt ---")
        print(AE_LANG_LLM_CODING_PROMPT)

    def print_script_control_prompt(self):
        print("\n--- AE-Lang LLM Script Control Prompt ---")
        print(AE_LANG_LLM_SCRIPT_CONTROL_PROMPT)

# === CLI ENTRYPOINT ===

def main():
    print("===== AE-Lang Interpreter (ILEICES) v2.0 Recursive =====")
    interpreter = AELangInterpreter()
    while True:
        print("\nMenu:")
        print(" 1. Run .ael script file")
        print(" 2. Paste AE-Lang script (multiline)")
        print(" 3. Print LLM Coding Prompt")
        print(" 4. Print LLM Script Control Prompt")
        print(" 5. Exit")
        choice = input("Select: ").strip()
        if choice == "1":
            filename = input("Enter .ael filename: ").strip()
            try:
                code = interpreter.load_ael_file(filename)
                interpreter.parse_script(code)
                cycles = int(input("Number of cycles to run (default 5): ") or "5")
                interpreter.run_recursive(cycles)
            except Exception as e:
                print(f"Error: {e}")
        elif choice == "2":
            print("Paste AE-Lang script, then enter a blank line to finish:")
            lines = []
            while True:
                l = input()
                if not l.strip(): break
                lines.append(l)
            code = "\n".join(lines)
            interpreter.parse_script(code)
            cycles = int(input("Number of cycles to run (default 5): ") or "5")
            interpreter.run_recursive(cycles)
        elif choice == "3":
            interpreter.print_coding_prompt()
        elif choice == "4":
            interpreter.print_script_control_prompt()
        elif choice == "5":
            print("Exiting AE-Lang Interpreter.")
            break
        else:
            print("Invalid choice.")

if __name__ == "__main__":
    main()
