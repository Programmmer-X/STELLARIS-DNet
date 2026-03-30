
# my function

import os

def setup_git():
    """
    Run this once per Colab session.
    Sets Git identity and enables credential storage.
    """
    os.system('git config --global user.email "your_email@example.com"')
    os.system('git config --global user.name "Your Name"')
    os.system('git config --global credential.helper store')

    print("✅ Git configured. You will be asked for token only once during first push.")


def git_push(commit_msg="update"):
    """
    Adds, commits, and pushes changes to GitHub.
    """
    # Add all changes
    os.system("git add .")

    # Commit
    os.system(f'git commit -m "{commit_msg}"')

    # Push (will ask token only first time)
    os.system("git push origin main")

    print("🚀 Changes pushed to GitHub successfully!")
