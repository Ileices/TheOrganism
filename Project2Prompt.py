# -*- coding: utf-8 -*-

"""

# Project2Prompt Application Documentation

## Table of Contents

1. [Introduction](#introduction)
2. [Features](#features)
3. [Installation](#installation)
4. [Running the Application](#running-the-application)
5. [User Interface Overview](#user-interface-overview)
6. [Panel Descriptions and Functionalities](#panel-descriptions-and-functionalities)
    - [Panel 1: Instruction Panel](#panel-1-instruction-panel)
    - [Panel 2: Final Prompt Assembly](#panel-2-final-prompt-assembly)
    - [Panel 3: ChatGPT Response Area](#panel-3-chatgpt-response-area)
    - [Panel 4: GUI Prompt Selector](#panel-4-gui-prompt-selector)
    - [Panel 5: Prior Continuity](#panel-5-prior-continuity)
    - [Panel 6: Edge Case & Debugging](#panel-6-edge-case--debugging)
    - [Panel 7: Future Continuity](#panel-7-future-continuity)
    - [Panel 8: Script Unifier Prompts](#panel-8-script-unifier-prompts)
    - [Panel 9: ChatGPT Reminder Prompts](#panel-9-chatgpt-reminder-prompts)
    - [Panel 10: Documentation Prompt Selector](#panel-10-documentation-prompt-selector)
    - [Panel 11: Path Updater Prompt Selector](#panel-11-path-updater-prompt-selector)
    - [Panel 12: Project Overview](#panel-12-project-overview)
7. [Buttons and Their Functionalities](#buttons-and-their-functionalities)
8. [Customization Options](#customization-options)
    - [Layout Customization](#layout-customization)
    - [Theme Settings](#theme-settings)
    - [Directory Setup](#directory-setup)
9. [Data Persistence and Project Management](#data-persistence-and-project-management)
    - [Saving a Project](#saving-a-project)
    - [Loading a Project](#loading-a-project)
    - [Auto-Save Functionality](#auto-save-functionality)
10. [Error Handling and Troubleshooting](#error-handling-and-troubleshooting)
11. [Frequently Asked Questions (FAQs)](#frequently-asked-questions-faqs)
12. [Future Enhancements](#future-enhancements)
13. [Support and Contact](#support-and-contact)

---

## Introduction

**Project2Prompt** is a comprehensive Python application designed to streamline and enhance your project development workflow by leveraging the capabilities of ChatGPT. Utilizing a PyQt5-based graphical user interface (GUI), Project2Prompt offers a suite of interactive panels that facilitate the assembly, management, and refinement of project prompts and responses. The application emphasizes automation, data persistence, and user customization to maximize efficiency and minimize manual input.

## Features

- **12 Interactive Panels:** Each panel serves a unique purpose, from instruction parsing to project overview.
- **Auto-Update Mechanism:** Panels dynamically update based on user inputs and ChatGPT responses.
- **Data Persistence:** Automatically saves project data to a local SQLite database and allows manual saving/loading via JSON files.
- **Customizable Layouts:** Users can rearrange panels using predefined layout styles.
- **Theme Settings:** Switch between Dark and Light themes for enhanced readability and aesthetics.
- **Clickable Elements:** Steps and links are interactive, enabling seamless navigation and integration.
- **Error Handling:** Robust mechanisms to handle unexpected inputs and ensure application stability.
- **Expandable Architecture:** Modular design allows for easy future enhancements and feature integrations.

## Installation

### Prerequisites

- **Python 3.6 or higher**: Ensure Python is installed on your system. You can download it from [Python's official website](https://www.python.org/downloads/).
- **pip**: Python's package installer, typically included with Python installations.

### Step-by-Step Installation

1. **Clone the Repository or Download the Script:**

   - **Clone via Git:**
     ```bash
     git clone https://github.com/your-repository/Project2Prompt.git
     ```
   - **Download Directly:**
     - Navigate to the repository and download the `Project2Prompt.py` script.

2. **Navigate to the Project Directory:**
   
   ```bash
   cd Project2Prompt
   ```

3. **Install Required Dependencies:**
   
   Install the necessary Python libraries using `pip`:
   
   ```bash
   pip install PyQt5 spacy pyqtgraph jsonschema rich prompt_toolkit pytest pylint faker
   ```

4. **Initialize the spaCy English Model:**
   
   The application utilizes spaCy for natural language processing. Initialize the English model with:
   
   ```bash
   python -m spacy download en_core_web_sm
   ```

## Running the Application

1. **Ensure All Dependencies Are Installed:**
   
   Double-check that all required libraries are installed. If you encounter any missing module errors, install them using `pip`.

2. **Execute the Script:**
   
   Run the application using Python:
   
   ```bash
   python Project2Prompt.py
   ```

   Alternatively, if you have multiple versions of Python installed:
   
   ```bash
   python3 Project2Prompt.py
   ```

3. **Accessing the Application:**
   
   Upon successful execution, the **Project2Prompt** window will appear, displaying all 12 panels and control buttons.

## User Interface Overview

The application window is divided into three main sections:

- **Left Pane:** Contains Panel 1 (Instruction Panel).
- **Middle Pane:** Houses Panel 2 (Final Prompt Assembly) and Panels 4-12 arranged in a grid layout.
- **Right Pane:** Features Panel 3 (ChatGPT Response Area).

At the bottom of Panel 2, a row of control buttons allows users to perform actions such as copying content, saving/loading projects, previewing prompts, customizing layouts, and accessing settings.

## Panel Descriptions and Functionalities

### Panel 1: Instruction Panel

**Purpose:**
- Serve as the starting point for project development by allowing users to input or load development plans.

**Functionalities:**
- **Input:** Users can paste or load a development plan formatted with discrete steps.
- **Auto-Parsing:** The panel automatically parses instructions into individual, clickable steps.
- **Interaction:** Clicking on a step auto-populates Panel 2 with the selected prompt, streamlining prompt assembly.

**Usage:**
1. **Pasting Instructions:**
   - Enter your development plan in the text area, ensuring each step is prefixed with "Step:" (e.g., "Step: Design the user interface").
2. **Parsing Steps:**
   - The application automatically detects and formats each step.
3. **Selecting Steps:**
   - Click on any step to transfer its content to Panel 2 for prompt assembly.

### Panel 2: Final Prompt Assembly

**Purpose:**
- Central hub for assembling, refining, and managing the final prompt to be used with ChatGPT.

**Functionalities:**
- **Dynamic Assembly:** Combines inputs from Panels 1, 4–12 to form a cohesive prompt.
- **Manual Overrides:** Users can manually edit the assembled prompt, with manual edits visually distinguished.
- **Clickable Link:** Contains a clickable link that opens a specified YouTube channel in the default browser.

**Usage:**
1. **Viewing Assembled Prompt:**
   - Automatically populated based on inputs from other panels.
2. **Editing Prompt:**
   - Toggle between manual and automatic update modes using the "Manual Updates" button.
   - Manual edits are visually distinct, aiding in differentiation from auto-generated content.
3. **Copying Prompt:**
   - Use the "Copy All" button to copy the entire prompt to the clipboard for use with ChatGPT.

### Panel 3: ChatGPT Response Area

**Purpose:**
- Area where users paste responses generated by ChatGPT based on the assembled prompt.

**Functionalities:**
- **Input:** Users can paste ChatGPT's responses.
- **Auto-Parsing:** The application parses the response to extract relevant data for other panels, such as next steps, errors, and completed steps.
- **Updates:** Automatically updates Panels 5, 6, 7, 8, and 9 based on parsed content.

**Usage:**
1. **Pasting Responses:**
   - After obtaining a response from ChatGPT, paste it into the text area.
2. **Automatic Parsing:**
   - The application processes the response, extracting information to update other panels.
3. **Reviewing Parsed Data:**
   - Check Panels 5, 6, 7, 8, and 9 for auto-populated content based on the response.

### Panel 4: GUI Prompt Selector

**Purpose:**
- Manage and select GUI-related prompts to be included in the final assembly.

**Functionalities:**
- **Selection:** Choose from predefined GUI prompts or create custom ones.
- **Integration:** Selected prompts are automatically added to Panel 2.

**Usage:**
1. **Selecting Prompts:**
   - Browse through available GUI prompts and select the ones relevant to your project.
2. **Adding to Prompt Assembly:**
   - Selected prompts are automatically integrated into Panel 2 for assembly.

### Panel 5: Prior Continuity

**Purpose:**
- Maintain and reference previously completed steps, ensuring project continuity.

**Functionalities:**
- **Auto-Update:** Populates references to completed steps and relevant scripts parsed from Panel 3.
- **Manual Adjustments:** Users can manually add or refine continuity context.
- **Integration:** Adds relevant continuity information to Panel 2.

**Usage:**
1. **Reviewing Continuity:**
   - Check for references to completed steps and ensure all necessary contexts are included.
2. **Adding References:**
   - Manually input any additional continuity information as needed.
3. **Ensuring Continuity in Prompt:**
   - Continuity information is automatically incorporated into Panel 2.

### Panel 6: Edge Case & Debugging

**Purpose:**
- Log and manage errors or unhandled scenarios to aid in debugging and refining prompts.

**Functionalities:**
- **Auto-Update:** Logs errors or inconsistencies detected in Panel 3's ChatGPT responses.
- **Manual Logging:** Allows users to manually record any additional edge cases or debugging information.
- **Integration:** Provides debugging prompts to Panel 2.

**Usage:**
1. **Reviewing Errors:**
   - Monitor for automatically logged errors from ChatGPT responses.
2. **Logging Additional Issues:**
   - Manually add any unhandled scenarios or specific debugging notes.
3. **Refining Prompts:**
   - Use the logged information to adjust and improve prompts in Panel 2.

### Panel 7: Future Continuity

**Purpose:**
- Outline and manage upcoming tasks and dependencies to ensure seamless project progression.

**Functionalities:**
- **Auto-Update:** Populates next steps parsed from Panel 3's ChatGPT responses.
- **Manual Adjustments:** Users can manually add or adjust future tasks and dependencies.
- **Integration:** Highlights upcoming tasks to be included in Panel 2.

**Usage:**
1. **Reviewing Next Steps:**
   - Check for automatically populated next steps based on ChatGPT's responses.
2. **Adding Upcoming Tasks:**
   - Manually input any additional tasks or dependencies as needed.
3. **Ensuring Progression:**
   - Upcoming tasks are automatically incorporated into Panel 2 for prompt assembly.

### Panel 8: Script Unifier Prompts

**Purpose:**
- Identify and unify dependencies across multiple scripts or modules to maintain project modularity.

**Functionalities:**
- **Auto-Update:** Analyzes Panel 3's data to identify dependencies and unify script logic.
- **Integration:** Provides unification prompts to Panel 2.

**Usage:**
1. **Reviewing Dependencies:**
   - Monitor for identified dependencies across scripts or modules.
2. **Unifying Scripts:**
   - Utilize the unification prompts to ensure scripts work cohesively.
3. **Incorporating into Prompt:**
   - Unification prompts are automatically added to Panel 2.

### Panel 9: ChatGPT Reminder Prompts

**Purpose:**
- Log and manage reminders or suggestions to enhance prompt quality and project continuity.

**Functionalities:**
- **Auto-Update:** Extracts reminders or suggestions from Panel 3's ChatGPT responses.
- **Manual Entries:** Users can add additional reminders as needed.
- **Integration:** Sends reminders to Panel 2 for better continuity.

**Usage:**
1. **Reviewing Reminders:**
   - Check for automatically extracted reminders to improve prompts.
2. **Adding Reminders:**
   - Manually input any additional reminders or suggestions.
3. **Enhancing Prompts:**
   - Reminders are integrated into Panel 2 to refine the final prompt.

### Panel 10: Documentation Prompt Selector

**Purpose:**
- Manage and integrate documentation-related prompts to ensure thorough project documentation.

**Functionalities:**
- **Auto-Update:** Integrates documentation information from Panel 3.
- **Manual Inputs:** Users can write or load additional documentation.
- **Integration:** Ensures documentation prompts are included in Panel 2.

**Usage:**
1. **Managing Documentation:**
   - Review automatically integrated documentation prompts.
2. **Adding Documentation:**
   - Manually input or load additional documentation as needed.
3. **Ensuring Documentation in Prompt:**
   - Documentation prompts are seamlessly added to Panel 2.

### Panel 11: Path Updater Prompt Selector

**Purpose:**
- Manage and update file paths detected within project data to maintain accurate references.

**Functionalities:**
- **Auto-Update:** Detects and updates file paths from Panel 3's ChatGPT responses.
- **Manual Updates:** Users can manually add or adjust file paths.
- **Integration:** Ensures accurate paths are included in Panel 2's prompt.

**Usage:**
1. **Reviewing File Paths:**
   - Check for automatically detected and updated file paths.
2. **Adding/Adjusting Paths:**
   - Manually input any additional or corrected file paths.
3. **Maintaining Accuracy:**
   - Updated paths are integrated into Panel 2 to ensure prompt accuracy.

### Panel 12: Project Overview

**Purpose:**
- Provide a comprehensive summary of the project's current state, including completed steps, active tasks, error logs, and key references.

**Functionalities:**
- **Auto-Update:** Aggregates data from Panels 1–11 to present an overview.
- **Manual Review:** Users can view and copy the project summary.
- **Separate Window:** Clicking the panel title opens the overview in a separate window for easy access.

**Usage:**
1. **Viewing Project Summary:**
   - Click on the title of Panel 12 to open a detailed overview in a new window.
2. **Reviewing Key Information:**
   - The overview includes completed steps, active tasks, error logs, and key references.
3. **Exporting Summary:**
   - Use the overview window to copy or export the project summary as needed.

## Buttons and Their Functionalities

Located at the bottom of Panel 2, the following buttons provide essential controls for managing your project:

1. **Copy All:**
   - **Function:** Copies the entire content of Panel 2 (Final Prompt Assembly) to the clipboard.
   - **Usage:** Click to quickly copy the assembled prompt for use with ChatGPT or other applications.

2. **Save Project:**
   - **Function:** Saves the current state of all panels into a JSON file.
   - **Usage:** Click to export your project data, enabling you to resume work later or share the project setup.

3. **Load Project:**
   - **Function:** Loads a previously saved project from a JSON file.
   - **Usage:** Click to import project data, restoring the state of all panels to a saved configuration.

4. **Preview:**
   - **Function:** Opens a dialog displaying the content of Panel 2 in a read-only format.
   - **Usage:** Click to review the final prompt without making edits, ensuring everything is correctly assembled.

5. **Directory Setup:**
   - **Function:** Opens a dialog to select and set the directory where project-related files will be stored.
   - **Usage:** Click to change the default save/load directory for projects and related files.

6. **Customize:**
   - **Function:** Opens a dialog allowing users to rearrange the layout of Panels 4-12.
   - **Usage:** Select a preferred layout style (Default, Reversed, ZigZag) and apply it to customize the panel arrangement.

7. **Settings:**
   - **Function:** Opens the settings dialog to configure application preferences, including theme selection and auto-save toggles.
   - **Usage:** Click to adjust visual themes (Dark/Light) and manage auto-save settings for enhanced user experience.

## Customization Options

### Layout Customization

**Purpose:**
- Allow users to rearrange Panels 4-12 to suit their workflow preferences.

**Available Layout Styles:**
1. **Default:**
   - Panels are arranged in the standard 3x3 grid as initially designed.

2. **Reversed:**
   - Panels are displayed in reverse order, enhancing different perspectives or workflow approaches.

3. **ZigZag:**
   - Panels alternate direction in each row, creating a zigzag pattern for varied visual arrangements.

**How to Customize:**
1. **Access Customize Dialog:**
   - Click the "Customize" button located at the bottom of Panel 2.

2. **Select Layout Style:**
   - In the dialog, choose your preferred layout style from the dropdown menu.

3. **Apply Layout:**
   - Click the "Apply Layout" button to rearrange Panels 4-12 according to your selection.

### Theme Settings

**Purpose:**
- Enhance readability and reduce eye strain by switching between Dark and Light themes.

**Available Themes:**
1. **Dark Theme:**
   - Default theme with dark backgrounds and green text, providing a modern and sleek appearance.

2. **Light Theme:**
   - Alternative theme with light backgrounds and dark text, offering a brighter interface.

**How to Change Theme:**
1. **Access Settings Dialog:**
   - Click the "Settings" button located at the bottom of Panel 2.

2. **Select Theme:**
   - In the dialog, choose "Dark" or "Light" from the "Theme" dropdown menu.

3. **Apply Theme:**
   - Click the "Save Settings" button to apply the selected theme across the application.

**Note:** Theme changes are applied immediately, providing instant visual feedback.

### Directory Setup

**Purpose:**
- Define the directory where project-related files (e.g., saved projects) will be stored.

**How to Set Up Directory:**
1. **Access Directory Setup Dialog:**
   - Click the "Directory Setup" button located at the bottom of Panel 2.

2. **Select Directory:**
   - In the dialog, click the "Browse" button to navigate to your desired directory.

3. **Confirm Selection:**
   - Once the directory is selected, its path will appear in the text field. Click outside the dialog or press "OK" to confirm.

**Benefits:**
- Organizes project files in a centralized location.
- Facilitates easy access and management of saved projects.

## Data Persistence and Project Management

### Saving a Project

**Purpose:**
- Preserve the current state of all panels, enabling you to resume work later or share your project setup.

**How to Save a Project:**
1. **Click "Save Project":**
   - Located at the bottom of Panel 2.

2. **Choose Save Location:**
   - In the file dialog, navigate to your desired save directory (as set in Directory Setup).

3. **Name Your Project:**
   - Enter a descriptive name for your project file (e.g., `MyProject.json`).

4. **Save as JSON:**
   - Ensure the file type is set to "JSON Files (*.json)" and click "Save."

5. **Confirmation:**
   - A message box will confirm that the project has been saved successfully.

**Note:**
- All panel contents, including auto-updated and manually edited data, are saved.

### Loading a Project

**Purpose:**
- Restore a previously saved project, repopulating all panels with the stored data.

**How to Load a Project:**
1. **Click "Load Project":**
   - Located at the bottom of Panel 2.

2. **Navigate to Project File:**
   - In the file dialog, go to the directory where your project JSON file is saved.

3. **Select Project File:**
   - Choose the desired JSON file (e.g., `MyProject.json`) and click "Open."

4. **Project Loaded:**
   - All panels will be populated with the data from the selected project file.
   - A confirmation message will appear upon successful loading.

**Note:**
- Loading a project replaces the current panel contents with the loaded data.

### Auto-Save Functionality

**Purpose:**
- Automatically save project data at regular intervals or upon specific triggers, minimizing data loss.

**Current Status:**
- **Implementation Pending:** The auto-save toggle is present in the Settings dialog but requires further development to function as intended.

**Future Enhancements:**
- Implement auto-save intervals (e.g., every 5 minutes).
- Ensure auto-saved data is stored securely and can be recovered in case of unexpected shutdowns.

**User Instructions:**
1. **Access Settings Dialog:**
   - Click the "Settings" button at the bottom of Panel 2.

2. **Toggle Auto-Save:**
   - Enter "on" or "off" in the "Toggle Auto-Save" field or use a checkbox (depending on future implementation).

3. **Save Settings:**
   - Click "Save Settings" to apply the auto-save preference.

**Note:**
- Currently, manual saving/loading is the primary method for data persistence.

## Customization Options

### Layout Customization

**Purpose:**
- Allow users to rearrange Panels 4-12 to suit their workflow preferences.

**Available Layout Styles:**
1. **Default:**
   - Panels are arranged in the standard 3x3 grid as initially designed.

2. **Reversed:**
   - Panels are displayed in reverse order, enhancing different perspectives or workflow approaches.

3. **ZigZag:**
   - Panels alternate direction in each row, creating a zigzag pattern for varied visual arrangements.

**How to Customize:**
1. **Access Customize Dialog:**
   - Click the "Customize" button located at the bottom of Panel 2.

2. **Select Layout Style:**
   - In the dialog, choose your preferred layout style from the dropdown menu.

3. **Apply Layout:**
   - Click the "Apply Layout" button to rearrange Panels 4-12 according to your selection.

**Benefits:**
- Enhances user experience by allowing personalized panel arrangements.
- Facilitates better workflow management based on individual project needs.

### Theme Settings

**Purpose:**
- Enhance readability and reduce eye strain by switching between Dark and Light themes.

**Available Themes:**
1. **Dark Theme:**
   - Default theme with dark backgrounds and green text, providing a modern and sleek appearance.

2. **Light Theme:**
   - Alternative theme with light backgrounds and dark text, offering a brighter interface.

**How to Change Theme:**
1. **Access Settings Dialog:**
   - Click the "Settings" button located at the bottom of Panel 2.

2. **Select Theme:**
   - In the dialog, choose "Dark" or "Light" from the "Theme" dropdown menu.

3. **Apply Theme:**
   - Click the "Save Settings" button to apply the selected theme across the application.

**Note:**
- Theme changes are applied immediately, providing instant visual feedback.

### Directory Setup

**Purpose:**
- Define the directory where project-related files (e.g., saved projects) will be stored.

**How to Set Up Directory:**
1. **Access Directory Setup Dialog:**
   - Click the "Directory Setup" button located at the bottom of Panel 2.

2. **Select Directory:**
   - In the dialog, click the "Browse" button to navigate to your desired directory.

3. **Confirm Selection:**
   - Once the directory is selected, its path will appear in the text field. Click outside the dialog or press "OK" to confirm.

**Benefits:**
- Organizes project files in a centralized location.
- Facilitates easy access and management of saved projects.

## Data Persistence and Project Management

### Saving a Project

**Purpose:**
- Preserve the current state of all panels, enabling you to resume work later or share the project setup.

**How to Save a Project:**
1. **Click "Save Project":**
   - Located at the bottom of Panel 2.

2. **Choose Save Location:**
   - In the file dialog, navigate to your desired save directory (as set in Directory Setup).

3. **Name Your Project:**
   - Enter a descriptive name for your project file (e.g., `MyProject.json`).

4. **Save as JSON:**
   - Ensure the file type is set to "JSON Files (*.json)" and click "Save."

5. **Confirmation:**
   - A message box will confirm that the project has been saved successfully.

**Note:**
- All panel contents, including auto-updated and manually edited data, are saved.

### Loading a Project

**Purpose:**
- Restore a previously saved project, repopulating all panels with the stored data.

**How to Load a Project:**
1. **Click "Load Project":**
   - Located at the bottom of Panel 2.

2. **Navigate to Project File:**
   - In the file dialog, go to the directory where your project JSON file is saved.

3. **Select Project File:**
   - Choose the desired JSON file (e.g., `MyProject.json`) and click "Open."

4. **Project Loaded:**
   - All panels will be populated with the data from the selected project file.
   - A confirmation message will appear upon successful loading.

**Note:**
- Loading a project replaces the current panel contents with the loaded data.

### Auto-Save Functionality

**Purpose:**
- Automatically save project data at regular intervals or upon specific triggers, minimizing data loss.

**Current Status:**
- **Implementation Pending:** The auto-save toggle is present in the Settings dialog but requires further development to function as intended.

**Future Enhancements:**
- Implement auto-save intervals (e.g., every 5 minutes).
- Ensure auto-saved data is stored securely and can be recovered in case of unexpected shutdowns.

**User Instructions:**
1. **Access Settings Dialog:**
   - Click the "Settings" button at the bottom of Panel 2.

2. **Toggle Auto-Save:**
   - Enter "on" or "off" in the "Toggle Auto-Save" field or use a checkbox (depending on future implementation).

3. **Save Settings:**
   - Click "Save Settings" to apply the auto-save preference.

**Note:**
- Currently, manual saving/loading is the primary method for data persistence.

## Error Handling and Troubleshooting

**Common Issues and Solutions:**

1. **Application Fails to Launch:**
   - **Symptom:** Error messages upon executing `Project2Prompt.py`.
   - **Solution:** Ensure all dependencies are installed correctly. Run `pip install -r requirements.txt` if a requirements file is provided or individually install missing packages.

2. **AttributeError: Missing Methods:**
   - **Symptom:** Errors indicating missing methods (e.g., `process_panel_text`).
   - **Solution:** Ensure the complete and correct version of `Project2Prompt.py` is being used. Verify that all methods are defined within the `Project2Prompt` class.

3. **Layout Issues:**
   - **Symptom:** Panels overlapping or not displaying correctly.
   - **Solution:** Ensure that each widget has only one layout. Review the layout setup in the code to prevent multiple layouts from being assigned to the same widget.

4. **Data Not Saving/Loading:**
   - **Symptom:** Projects are not being saved or loaded as expected.
   - **Solution:** Verify that the application has write permissions to the selected save directory. Check for any error messages during save/load operations and address them accordingly.

5. **Theme Not Applying:**
   - **Symptom:** Selected theme does not change the application's appearance.
   - **Solution:** Ensure that the theme selection is correctly implemented in the `apply_dark_theme` and `apply_light_theme` methods. Restart the application if necessary.

6. **Clickable Links Not Functioning:**
   - **Symptom:** Clicking on the link text does not open the browser.
   - **Solution:** Confirm that the link detection logic in `detect_link_click` is correctly identifying the selected text. Ensure that the system's default browser is properly configured.

**General Troubleshooting Steps:**

- **Check Dependencies:** Ensure all required Python libraries are installed.
- **Review Error Messages:** Read and understand error outputs to identify problematic areas in the code.
- **Consult Documentation:** Refer to this documentation for guidance on functionalities and settings.
- **Update Application:** Ensure that you are using the latest version of `Project2Prompt.py`.
- **Seek Support:** If issues persist, consider reaching out to support channels or forums for assistance.

## Frequently Asked Questions (FAQs)

1. **How do I add a new step in Panel 1?**
   - **Answer:** In Panel 1, enter your development plan with each step prefixed by "Step:". For example:
     ```
     Step: Design the user interface.
     Step: Implement the backend logic.
     Step: Integrate the database.
     ```
     The application will automatically parse and display each step as clickable items.

2. **Can I customize the colors and fonts beyond the provided themes?**
   - **Answer:** Currently, the application offers Dark and Light themes. Future updates may include more customization options for colors and fonts. For advanced customization, modify the stylesheet in the code as needed.

3. **Is it possible to export the project summary?**
   - **Answer:** Yes. Navigate to Panel 12 (Project Overview) and click on the panel title to open the overview in a separate window. From there, you can copy the summary or implement export functionalities as needed.

4. **How does the auto-update mechanism work?**
   - **Answer:** The application parses inputs from Panels 1 and 3 to automatically update related panels (Panels 4-12). For example, pasting a ChatGPT response in Panel 3 will extract next steps, errors, and other relevant information to update Panels 5-9 accordingly.

5. **Can I revert to a previous layout after customizing?**
   - **Answer:** Yes. Click the "Customize" button, select the desired layout style (e.g., Default), and apply it to rearrange the panels accordingly.

6. **How do I reset all panels to their default states?**
   - **Answer:** Manually clear the content of each panel or reload a previously saved default project using the "Load Project" button.

7. **Is my data stored securely?**
   - **Answer:** Project data is stored locally in a SQLite database (`project2prompt.db`) and can be exported/imported via JSON files. Ensure that your system's security measures are in place to protect these files.

8. **Can I integrate additional functionalities or panels?**
   - **Answer:** Yes. The application's modular design allows for the addition of new panels and features. Modify the code to include new components as needed.

## Future Enhancements

**Planned Features:**

1. **Advanced Parsing:**
   - Implement more sophisticated natural language processing to better interpret and categorize ChatGPT responses.

2. **Auto-Save Functionality:**
   - Develop an auto-save feature that periodically saves project data without manual intervention.

3. **Enhanced Settings:**
   - Introduce more customization options, including adjustable font sizes, color schemes, and additional themes.

4. **Input Validation:**
   - Integrate `jsonschema` to validate JSON files during save/load operations, ensuring data integrity.

5. **Automated Testing:**
   - Incorporate `pytest` and `pylint` for automated unit and integration testing, enhancing code reliability.

6. **User Interface Improvements:**
   - Add icons using `FontAwesome` and customizable fonts via `Google Fonts` to enhance the visual appeal of the application.

7. **Documentation Generation:**
   - Utilize tools like `MkDocs` or `Sphinx` to generate comprehensive user and developer documentation.

8. **Exporting Options:**
   - Enable exporting project summaries and prompts in various formats (e.g., PDF, DOCX).

9. **Collaboration Features:**
   - Introduce functionalities that allow multiple users to collaborate on the same project in real-time.

10. **Integration with Other Tools:**
    - Facilitate integration with project management tools like Trello or Jira for enhanced workflow management.

**User Feedback:**
- Your feedback is invaluable! Please share your suggestions and feature requests to help shape the future development of Project2Prompt.

## Support and Contact

For further assistance, bug reports, or feature requests, please reach out through the following channels:

- **Email:** support@project2prompt.com
- **GitHub Issues:** [Project2Prompt Repository Issues](https://github.com/your-repository/Project2Prompt/issues)
- **Community Forums:** [Project2Prompt Community](https://forum.project2prompt.com)

**Note:** Replace the placeholder links and contact information with actual details relevant to your project.

---

**Thank you for choosing Project2Prompt! We hope this documentation helps you make the most of the application. Happy project developing!**


"""

import sys
import os
import sqlite3
import json
import re
from functools import partial
from pathlib import Path

import spacy
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QGridLayout,
    QPushButton, QScrollArea, QTextEdit, QFileDialog, QLabel, QDialog,
    QLineEdit, QMessageBox, QComboBox, QTabWidget, QCheckBox
)
from PyQt5.QtCore import Qt, QRect, QUrl, QSize
from PyQt5.QtGui import QDesktopServices, QTextCursor, QFont, QIcon

# Initialize spaCy
try:
    nlp = spacy.load("en_core_web_sm")
except:
    import subprocess
    subprocess.run([sys.executable, "-m", "spacy", "download", "en_core_web_sm"])
    nlp = spacy.load("en_core_web_sm")

# Constants
APP_TITLE = "Project2Prompt"
DEFAULT_WINDOW_SIZE = (1080, 720)
LINK_TEXT = "The God Factory | Project Ileices | Roswan Miller"
LINK_URL = "https://www.youtube.com/channel/UCb0ZEDfL0Zg-2jjIY8zZ7SQ"

# Utility Functions
def resource_path(relative_path):
    """ Get absolute path to resource, works for dev and for PyInstaller """
    try:
        base_path = sys._MEIPASS
    except Exception:
        base_path = os.path.abspath(".")
    return os.path.join(base_path, relative_path)

# ScrollPanel Class
class ScrollPanel(QWidget):
    def __init__(self, panel_title, parent=None):
        super().__init__(parent)
        self.panel_title = panel_title
        self.init_ui()

    def init_ui(self):
        # Main layout for the panel
        self.main_layout = QVBoxLayout()
        self.setLayout(self.main_layout)

        # Title Label
        self.title_label = QLabel(self.panel_title)
        self.title_label.setAlignment(Qt.AlignCenter)
        self.title_label.setFont(QFont("Cascadia Code", 9, QFont.Bold))
        self.main_layout.addWidget(self.title_label)

        # Data Box
        self.data_box = QTextEdit()
        self.data_box.setLineWrapMode(QTextEdit.WidgetWidth)
        self.main_layout.addWidget(self.data_box)
        self.data_box.setReadOnly(False)  # Ensure the data box is editable

        # Toggle Button
        self.toggle_button = QPushButton("Manual Updates")
        self.toggle_button.setCheckable(True)
        self.toggle_button.clicked.connect(self.toggle_update_mode)
        self.main_layout.addWidget(self.toggle_button)

        self.setStyleSheet("""
            QMainWindow, QWidget {
                background-color: #000000;
                color: #00FF00;
            }
            QLabel {
                color: #00FF00;
                font-weight: bold;
            }
            QTextEdit {
                background-color: #222222;
                color: #00FF00;
                border: 2px solid rgba(255, 0, 0, 0.7);
            }
            QPushButton {
                background-color: #FF8C00;
                color: #000000;
                border: 2px solid rgba(255, 0, 0, 0.7);
                border-radius: 15px;
            }
            QPushButton:hover {
                background-color: #ffaa33;
            }
            QScrollArea {
                background-color: #000000;
                border: none;
            }
            QLineEdit {
                background-color: #222222;
                color: #00FF00;
                border: 2px solid rgba(255, 0, 0, 0.7);
            }
            QTabWidget::pane {
                background-color: #000000;
            }
            QScrollBar:vertical, QScrollBar:horizontal {
                background-color: #111111;
                border: none;
            }
            QScrollBar::handle {
                background-color: #333333;
                border-radius: 4px;
            }
            QScrollBar::add-line, QScrollBar::sub-line {
                background-color: #222222;
            }
            QScrollBar::add-page, QScrollBar::sub-page {
                background-color: none;
            }
        """)

    def toggle_update_mode(self, checked):
        if checked:
            self.toggle_button.setText("Automatic Updates")
            self.data_box.setStyleSheet("""
                QTextEdit {
                    background-color: #222222;
                    color: #00FF00;
                    border: 2px solid rgba(0, 255, 0, 0.7);
                }
            """)
        else:
            self.toggle_button.setText("Manual Updates")
            self.data_box.setStyleSheet("""
                QTextEdit {
                    background-color: #333333;
                    color: #00FF00;
                    border: 2px solid rgba(255, 0, 0, 0.7);
                }
            """)

# Main Application Class
class Project2Prompt(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle(APP_TITLE)
        self.resize(*DEFAULT_WINDOW_SIZE)
        self.setWindowIcon(QIcon(resource_path("icon.png")))  # Optional: Add an icon if available

        # Initialize Database
        self.init_db()

        # Initialize UI
        self.init_ui()

        # Load Data
        self.load_data()

    def init_db(self):
        self.conn = sqlite3.connect("project2prompt.db")
        self.c = self.conn.cursor()
        self.c.execute("""
            CREATE TABLE IF NOT EXISTS panel_data (
                panel_id INTEGER PRIMARY KEY,
                content TEXT
            )
        """)
        self.conn.commit()

    def init_ui(self):
        # Central Widget
        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)
        self.main_layout = QHBoxLayout(self.central_widget)

        # Left, Middle, Right Layouts
        self.left_layout = QVBoxLayout()
        self.middle_layout = QVBoxLayout()
        self.right_layout = QVBoxLayout()

        self.main_layout.addLayout(self.left_layout, 1)
        self.main_layout.addLayout(self.middle_layout, 1)
        self.main_layout.addLayout(self.right_layout, 1)

        # Create Panels 1, 2, 3
        self.panel1 = ScrollPanel("Panel 1: Instruction Panel")
        self.panel2 = ScrollPanel("Panel 2: Final Prompt Assembly")
        self.panel3 = ScrollPanel("Panel 3: ChatGPT Response Area")

        # Adjust Panel 2 to be the same height as Panel 1 and position it next to Panel 1
        self.left_layout.addWidget(self.panel1)
        self.left_layout.addWidget(self.panel2)

        self.right_layout.addWidget(self.panel3)

        # Create Grid for Panels 4-12
        self.grid_widget = QWidget()
        self.grid_layout = QGridLayout(self.grid_widget)
        self.middle_layout.addWidget(self.grid_widget)

        self.panels = []
        panel_titles = [
            "Panel 4: GUI Prompt Selector",
            "Panel 5: Prior Continuity",
            "Panel 6: Edge Case & Debugging",
            "Panel 7: Future Continuity",
            "Panel 8: Script Unifier Prompts",
            "Panel 9: ChatGPT Reminder Prompts",
            "Panel 10: Documentation Prompt Selector",
            "Panel 11: Path Updater Prompt Selector",
            "Panel 12: Project Overview"
        ]

        for title in panel_titles:
            panel = ScrollPanel(title)
            self.panels.append(panel)

        # Arrange Panels 4-12 in 3x3 Grid
        positions = [(i, j) for i in range(3) for j in range(3)]
        for position, panel in zip(positions, self.panels):
            self.grid_layout.addWidget(panel, *position)

        # Button Row for Panel 2
        self.button_layout = QHBoxLayout()
        self.middle_layout.addLayout(self.button_layout)

        self.copy_all_btn = QPushButton("Copy All")
        self.save_project_btn = QPushButton("Save Project")
        self.load_project_btn = QPushButton("Load Project")
        self.preview_btn = QPushButton("Preview")
        self.directory_setup_btn = QPushButton("Directory Setup")
        self.customize_btn = QPushButton("Customize")
        self.settings_btn = QPushButton("Settings")

        self.button_layout.addWidget(self.copy_all_btn)
        self.button_layout.addWidget(self.save_project_btn)
        self.button_layout.addWidget(self.load_project_btn)
        self.button_layout.addWidget(self.preview_btn)
        self.button_layout.addWidget(self.directory_setup_btn)
        self.button_layout.addWidget(self.customize_btn)
        self.button_layout.addWidget(self.settings_btn)

        # Connect Buttons
        self.copy_all_btn.clicked.connect(self.copy_all)
        self.save_project_btn.clicked.connect(self.save_project)
        self.load_project_btn.clicked.connect(self.load_project)
        self.preview_btn.clicked.connect(self.show_preview)
        self.directory_setup_btn.clicked.connect(self.directory_setup)
        self.customize_btn.clicked.connect(self.customize_layout)
        self.settings_btn.clicked.connect(self.open_settings)

        # Setup Signals for Panels
        self.setup_panel_signals()

        # Add Copyright
        self.add_copyright()

    def setup_panel_signals(self):
        # Panel 1: Instruction Panel
        self.panel1.data_box.textChanged.connect(partial(self.process_panel_text, self.panel1, 1))

        # Panel 3: ChatGPT Response Area
        self.panel3.data_box.textChanged.connect(self.parse_chatgpt_response)

        # Panels 4-12: Specific Functionality
        for idx, panel in enumerate(self.panels, start=4):
            panel.data_box.textChanged.connect(partial(self.process_panel_text, panel, idx))

        # Panel 12: Open separate window on click
        self.panels[8].title_label.mousePressEvent = self.show_project_overview

    def process_panel_text(self, panel, panel_id):
        text = panel.data_box.toPlainText()
        # Implement parsing logic based on panel_id if needed
        # For example, extract steps and make them clickable
        # Here, we'll keep it simple
        pass

    def parse_chatgpt_response(self):
        text = self.panel3.data_box.toPlainText()
        doc = nlp(text)
        # Example: Parse and update Panels 5,6,7 based on keywords
        if "updatepanel5" in text.lower():
            self.update_panel(5, text)
        if "updatepanel6" in text.lower():
            self.update_panel(6, text)
        if "updatepanel7" in text.lower():
            self.update_panel(7, text)

    def update_panel(self, panel_number, content):
        index = panel_number - 4  # Panels 5-7 are at indices 1-3
        if 0 <= index < len(self.panels):
            current_text = self.panels[index].data_box.toPlainText()
            updated_text = current_text + "\n" + content
            self.panels[index].data_box.setPlainText(updated_text)

    def copy_all(self):
        QApplication.clipboard().setText(self.panel2.data_box.toPlainText())
        QMessageBox.information(self, "Copied", "All content copied to clipboard.")

    def save_project(self):
        project_data = {}
        # Save Panel1, Panel2, Panel3
        project_data["1"] = self.panel1.data_box.toPlainText()
        project_data["2"] = self.panel2.data_box.toPlainText()
        project_data["3"] = self.panel3.data_box.toPlainText()
        # Save Panels 4-12
        for idx, panel in enumerate(self.panels, start=4):
            project_data[str(idx)] = panel.data_box.toPlainText()

        options = QFileDialog.Options()
        file_name, _ = QFileDialog.getSaveFileName(self, "Save Project", "",
                                                   "JSON Files (*.json);;All Files (*)", options=options)
        if file_name:
            try:
                with open(file_name, 'w', encoding='utf-8') as f:
                    json.dump(project_data, f, ensure_ascii=False, indent=4)
                QMessageBox.information(self, "Success", "Project saved successfully.")
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Failed to save project:\n{e}")

    def load_project(self):
        options = QFileDialog.Options()
        file_name, _ = QFileDialog.getOpenFileName(self, "Load Project", "",
                                                   "JSON Files (*.json);;All Files (*)", options=options)
        if file_name:
            try:
                with open(file_name, 'r', encoding='utf-8') as f:
                    project_data = json.load(f)
                # Load Panel1, Panel2, Panel3
                self.panel1.data_box.setPlainText(project_data.get("1", ""))
                self.panel2.data_box.setPlainText(project_data.get("2", ""))
                self.panel3.data_box.setPlainText(project_data.get("3", ""))
                # Load Panels 4-12
                for idx, panel in enumerate(self.panels, start=4):
                    panel.data_box.setPlainText(project_data.get(str(idx), ""))
                QMessageBox.information(self, "Success", "Project loaded successfully.")
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Failed to load project:\n{e}")

    def show_preview(self):
        preview_dialog = QDialog(self)
        preview_dialog.setWindowTitle("Prompt Preview")
        preview_dialog.resize(600, 400)
        layout = QVBoxLayout(preview_dialog)
        preview_text = QTextEdit()
        preview_text.setReadOnly(True)
        preview_text.setPlainText(self.panel2.data_box.toPlainText())
        layout.addWidget(preview_text)
        preview_dialog.exec_()

    def directory_setup(self):
        dialog = QDialog(self)
        dialog.setWindowTitle("Directory Setup")
        dialog.resize(400, 200)
        layout = QVBoxLayout(dialog)
        dir_label = QLabel("Select Directory for Project Files:")
        layout.addWidget(dir_label)
        dir_line_edit = QLineEdit()
        layout.addWidget(dir_line_edit)
        browse_btn = QPushButton("Browse")
        layout.addWidget(browse_btn)

        def browse():
            directory = QFileDialog.getExistingDirectory(self, "Select Directory", os.getcwd())
            if directory:
                dir_line_edit.setText(directory)

        browse_btn.clicked.connect(browse)
        dialog.exec_()

    def customize_layout(self):
        dialog = QDialog(self)
        dialog.setWindowTitle("Customize Layout")
        dialog.resize(300, 200)
        layout = QVBoxLayout(dialog)

        label = QLabel("Choose Layout Style:")
        layout.addWidget(label)

        combo = QComboBox()
        combo.addItems(["Default", "Reversed", "ZigZag"])
        layout.addWidget(combo)

        apply_btn = QPushButton("Apply Layout")
        layout.addWidget(apply_btn)

        def apply():
            choice = combo.currentText()
            self.rearrange_panels(choice)
            dialog.accept()

        apply_btn.clicked.connect(apply)
        dialog.exec_()

    def rearrange_panels(self, style):
        # Remove all panels from grid
        for i in reversed(range(self.grid_layout.count())):
            widget = self.grid_layout.itemAt(i).widget()
            if widget:
                self.grid_layout.removeWidget(widget)

        # Determine new order based on style
        if style == "Default":
            ordered_panels = self.panels
        elif style == "Reversed":
            ordered_panels = list(reversed(self.panels))
        elif style == "ZigZag":
            ordered_panels = []
            for i in range(3):
                row_panels = self.panels[i*3:(i+1)*3]
                if i % 2 == 1:
                    row_panels = list(reversed(row_panels))
                ordered_panels.extend(row_panels)
        else:
            ordered_panels = self.panels

        # Re-add panels to grid
        positions = [(i, j) for i in range(3) for j in range(3)]
        for position, panel in zip(positions, ordered_panels):
            self.grid_layout.addWidget(panel, *position)

    def open_settings(self):
        settings_dialog = QDialog(self)
        settings_dialog.setWindowTitle("Settings")
        settings_dialog.resize(300, 200)
        layout = QVBoxLayout(settings_dialog)

        theme_label = QLabel("Theme:")
        layout.addWidget(theme_label)
        theme_combo = QComboBox()
        theme_combo.addItems(["Dark", "Light"])
        layout.addWidget(theme_combo)

        autosave_checkbox = QCheckBox("Toggle Auto-Save")
        layout.addWidget(autosave_checkbox)

        save_btn = QPushButton("Save Settings")
        layout.addWidget(save_btn)

        def save_settings():
            # Implement settings saving logic
            theme = theme_combo.currentText()
            if theme == "Dark":
                self.apply_dark_theme()
            else:
                self.apply_light_theme()

            self.auto_save_enabled = autosave_checkbox.isChecked()
            QMessageBox.information(self, "Settings", "Settings saved successfully.")
            settings_dialog.accept()

        save_btn.clicked.connect(save_settings)
        settings_dialog.exec_()

    def apply_dark_theme(self):
        self.setStyleSheet("""
            QMainWindow, QWidget {
                background-color: #000000;
                color: #00FF00;
            }
            QLabel {
                color: #00FF00;
                font-weight: bold;
            }
            QTextEdit {
                background-color: #222222;
                color: #00FF00;
                border: 2px solid rgba(255, 0, 0, 0.7);
            }
            QPushButton {
                background-color: #FF8C00;
                color: #000000;
                border: 2px solid rgba(255, 0, 0, 0.7);
                border-radius: 15px;
            }
            QPushButton:hover {
                background-color: #ffaa33;
            }
            QScrollArea {
                background-color: #000000;
                border: none;
            }
            QLineEdit {
                background-color: #222222;
                color: #00FF00;
                border: 2px solid rgba(255, 0, 0, 0.7);
            }
            QTabWidget::pane {
                background-color: #000000;
            }
            QScrollBar:vertical, QScrollBar:horizontal {
                background-color: #111111;
                border: none;
            }
            QScrollBar::handle {
                background-color: #333333;
                border-radius: 4px;
            }
            QScrollBar::add-line, QScrollBar::sub-line {
                background-color: #222222;
            }
            QScrollBar::add-page, QScrollBar::sub-page {
                background-color: none;
            }
            QDialog {
                background-color: #000000;
                color: #00FF00;
            }
            QMessageBox {
                background-color: #000000;
                color: #00FF00;
            }
        """)

    def apply_light_theme(self):
        self.setStyleSheet("""
            QMainWindow, QWidget {
                background-color: #FFFFFF;
                color: #000000;
            }
            QLabel {
                color: #000000;
                font-weight: bold;
            }
            QTextEdit {
                background-color: #FFFFFF;
                color: #000000;
                border: 2px solid rgba(0, 0, 0, 0.7);
            }
            QPushButton {
                background-color: #FF8C00;
                color: #000000;
                border: 2px solid rgba(0, 0, 0, 0.7);
                border-radius: 15px;
            }
            QPushButton:hover {
                background-color: #ffaa33;
            }
            QScrollArea {
                background-color: #FFFFFF;
                border: none;
            }
            QLineEdit {
                background-color: #FFFFFF;
                color: #000000;
                border: 2px solid rgba(0, 0, 0, 0.7);
            }
            QTabWidget::pane {
                background-color: #FFFFFF;
            }
            QScrollBar:vertical, QScrollBar:horizontal {
                background-color: #DDDDDD;
                border: none;
            }
            QScrollBar::handle {
                background-color: #CCCCCC;
                border-radius: 4px;
            }
            QScrollBar::add-line, QScrollBar::sub-line {
                background-color: #BBBBBB;
            }
            QScrollBar::add-page, QScrollBar::sub-page {
                background-color: none;
            }
            QDialog {
                background-color: #FFFFFF;
                color: #000000;
            }
            QMessageBox {
                background-color: #FFFFFF;
                color: #000000;
            }
        """)

    def show_project_overview(self, event=None):
        overview_dialog = QDialog(self)
        overview_dialog.setWindowTitle("Project Overview")
        overview_dialog.resize(400, 300)
        layout = QVBoxLayout(overview_dialog)
        overview_text = QTextEdit()
        overview_text.setReadOnly(True)
        overview_text.setPlainText(self.panels[8].data_box.toPlainText())
        layout.addWidget(overview_text)
        overview_dialog.exec_()

    def add_clickable_steps(self, panel, steps):
        for step in steps:
            cursor = panel.data_box.textCursor()
            cursor.movePosition(QTextCursor.End)
            cursor.insertText(f"Step: {step}\n")
            # Additional logic to make steps clickable can be added here

    def add_clickable_step(self, panel, step_text):
        cursor = panel.data_box.textCursor()
        cursor.movePosition(QTextCursor.End)
        cursor.insertText(f"Step: {step_text}\n")
        # Optionally, implement clickable functionality here

    def add_copyright(self):
        cursor = self.panel2.data_box.textCursor()
        cursor.movePosition(QTextCursor.End)
        cursor.insertText(f"\n{LINK_TEXT}\n")
        # Connect selection to open link
        self.panel2.data_box.cursorPositionChanged.connect(self.detect_link_click)

    def detect_link_click(self):
        cursor = self.panel2.data_box.textCursor()
        if cursor.hasSelection():
            selected_text = cursor.selectedText()
            if LINK_TEXT in selected_text:
                self.open_link()

    def open_link(self):
        QDesktopServices.openUrl(QUrl(LINK_URL))

    def load_data(self):
        # Load data from SQLite if needed
        self.c.execute("SELECT panel_id, content FROM panel_data")
        rows = self.c.fetchall()
        for row in rows:
            panel_id, content = row
            if panel_id == 1:
                self.panel1.data_box.setPlainText(content)
            elif panel_id == 2:
                self.panel2.data_box.setPlainText(content)
            elif panel_id == 3:
                self.panel3.data_box.setPlainText(content)
            elif 4 <= panel_id <= 12:
                self.panels[panel_id - 4].data_box.setPlainText(content)

    def closeEvent(self, event):
        # Save data to SQLite on close
        self.c.execute("DELETE FROM panel_data")
        # Save Panel1, Panel2, Panel3
        self.c.execute("INSERT INTO panel_data (panel_id, content) VALUES (?, ?)", (1, self.panel1.data_box.toPlainText()))
        self.c.execute("INSERT INTO panel_data (panel_id, content) VALUES (?, ?)", (2, self.panel2.data_box.toPlainText()))
        self.c.execute("INSERT INTO panel_data (panel_id, content) VALUES (?, ?)", (3, self.panel3.data_box.toPlainText()))
        # Save Panels 4-12
        for idx, panel in enumerate(self.panels, start=4):
            self.c.execute("INSERT INTO panel_data (panel_id, content) VALUES (?, ?)", (idx, panel.data_box.toPlainText()))
        self.conn.commit()
        self.conn.close()
        super().closeEvent(event)

# Entry Point
def main():
    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    font = QFont("Arial", 10)
    app.setFont(font)

    # Apply a global stylesheet
    app.setStyleSheet("""
        QMainWindow, QWidget {
            background-color: #000000;
            color: #00FF00;
        }
        QLabel {
            color: #00FF00;
            font-weight: bold;
        }
        QTextEdit {
            background-color: #222222;
            color: #00FF00;
            border: 2px solid rgba(255, 0, 0, 0.7);
        }
        QPushButton {
            background-color: #FF8C00;
            color: #000000;
            border: 2px solid rgba(255, 0, 0, 0.7);
            border-radius: 15px;
        }
        QPushButton:hover {
            background-color: #ffaa33;
        }
        QScrollArea {
            background-color: #000000;
            border: none;
        }
        QLineEdit {
            background-color: #222222;
            color: #00FF00;
            border: 2px solid rgba(255, 0, 0, 0.7);
        }
        QTabWidget::pane {
            background-color: #000000;
        }
        QScrollBar:vertical, QScrollBar:horizontal {
            background-color: #111111;
            border: none;
        }
        QScrollBar::handle {
            background-color: #333333;
            border-radius: 4px;
        }
        QScrollBar::add-line, QScrollBar::sub-line {
            background-color: #222222;
        }
        QScrollBar::add-page, QScrollBar::sub-page {
            background-color: none;
        }
        QDialog {
            background-color: #000000;
            color: #00FF00;
        }
        QMessageBox {
            background-color: #000000;
            color: #00FF00;
        }
    """)

    window = Project2Prompt()
    window.show()
    sys.exit(app.exec_())


def start_gui():
    main()

if __name__ == "__main__":
    start_gui()
