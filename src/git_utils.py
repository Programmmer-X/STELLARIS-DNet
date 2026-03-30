
# my function

import os
from getpass import getpass

def auto_push_notebook(repo_path="/content/STELLARIS-DNet",
                       notebook_name=None,
                       target_folder="notebooks/experiments",
                       commit_msg="Auto notebook update",
                       repo_url="https://github.com/Programmmer-X/STELLARIS-DNet.git"):

    TOKEN = getpass("Enter GitHub PAT: ")

    if notebook_name is None:
        print("⚠️ Please provide notebook_name manually")
        return

    nb_path = f"/content/{notebook_name}"
    target_path = f"{repo_path}/{target_folder}"

    os.makedirs(target_path, exist_ok=True)

    os.system(f'cp "{nb_path}" "{target_path}/"')

    os.chdir(repo_path)
    os.system("git add .")
    os.system(f'git commit -m "{commit_msg}"')
    os.system(f'git push https://{TOKEN}@{repo_url.replace("https://", "")} main')
    print("✅ Notebook pushed successfully!")
