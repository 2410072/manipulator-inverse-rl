
import os

SOURCE_PATH = '/home/shimoiyusuke/manipulator-inverse-rl/Compare/Final_Report.md'
DEST_PATH = '/home/shimoiyusuke/manipulator-inverse-rl/Compare/README.md'

def create_readme():
    print(f"Reading {SOURCE_PATH}...")
    with open(SOURCE_PATH, 'r', encoding='utf-8') as f:
        content = f.read()

    header = """# Project Report

**[📄 English Report (Original)](./Final_Report.md)** | **[📄 Japanese Report / 日本語レポート](./Final_Report_JP.md)**

---

"""
    
    full_content = header + content
    
    print(f"Writing to {DEST_PATH}...")
    with open(DEST_PATH, 'w', encoding='utf-8') as f:
        f.write(full_content)
    
    print("README.md created successfully.")

if __name__ == "__main__":
    create_readme()
