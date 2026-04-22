# How to create a virtual environment in python

Creating a virtual environment in Python allows you to manage project-specific dependencies separately, preventing version conflicts between different projects.

## 1. Create the Environment

Open your terminal or command prompt and navigate to your project folder. Run the following command to create a folder (conventionally named .venv or venv) containing the virtual environment

```bash
    Windows/macOS/Linux: python -m venv .venv
  (Note: Use python3 instead of python on some macOS/Linux systems if needed.)
```

## 2. Activate the Environment

You must "activate" the environment so your terminal knows to use that specific Python interpreter and its libraries.

- Windows (Command Prompt): **.venv\Scripts\activate**
- Windows (PowerShell): **.\.venv\Scripts\Activate.ps1**
- macOS/Linux: **source .venv/bin/activate**

Once active, you will usually see (.venv) appear at the start of your command prompt.

## 3. Usage & Deactivation

- Install Packages: While the environment is active, use pip install <package_name> to install libraries only for this project.
- Stop Using: When finished, simply type deactivate in the terminal to return to your global Python environment.
- Delete: To completely remove the environment, just delete the .venv folder from your project directory
