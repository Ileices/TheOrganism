# organism_cli.py - Chat interface for the Absolute Organism NLP Engine
#
# This script provides a simple command-line interface for interacting with the recursive_nlp_engine.py script.
# It allows users to input text and receive responses from the NLP engine.
# The chat interface runs in a loop, prompting the user for input and displaying the engine's response.
# The user can type 'exit' to quit the chat interface.
# The chat interface uses the subprocess module to run the recursive_nlp_engine.py script and capture its output.
# The output of the NLP engine is displayed to the user in the chat interface.
# The chat interface provides a convenient way to interact with the NLP engine and test its capabilities.
# The chat interface can be extended with additional features and functionality to enhance the user experience.
# The chat interface is a key component of the Absolute Organism project, enabling users to communicate with the AI system.
# The chat interface demonstrates the integration of the NLP engine into a user-friendly interface for real-time interaction.
# The chat interface is designed to be intuitive and easy to use, allowing users to engage with the AI system in a natural way.
# The chat interface leverages the power of the NLP engine to provide intelligent responses to user input, creating a dynamic conversational experience.
# The chat interface is an essential tool for exploring the capabilities of the NLP engine and unlocking its full potential.
# The chat interface is a valuable resource for developers, researchers, and enthusiasts who want to interact with the AI system and explore its capabilities.
# The chat interface is a key component of the Absolute Organism project, enabling users to engage with the AI system and experience its intelligence firsthand.
# The chat interface is a powerful tool for testing and refining the NLP engine, allowing developers to evaluate its performance and identify areas for improvement.
# The chat interface is an essential feature of the Absolute Organism project, providing a user-friendly interface for interacting with the AI system and exploring its capabilities.
# The chat interface is a versatile tool that can be customized and extended to meet the needs of different users and applications, making it a valuable asset for the Absolute Organism project.
# The chat interface is a key component of the Absolute Organism project, enabling users to interact with the AI system and experience its intelligence in a conversational setting.
# The chat interface is designed to be user-friendly and intuitive, allowing users to engage with the AI system in a natural and interactive way.




import subprocess
import sys

def chat_interface():
    print("\n🔹 Absolute Organism Chat Interface 🔹\n(Type 'exit' to quit)\n")
    while True:
        user_input = input("You: ")
        if user_input.lower() == 'exit':
            print("Exiting chat.")
            sys.exit()
        
        # ✅ FIX: Open recursive NLP engine in a **SEPARATE** CMD window
        subprocess.Popen(f"start cmd /k python recursive_nlp_engine.py \"{user_input}\"", shell=True)

if __name__ == "__main__":
    chat_interface()

