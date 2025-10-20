from langchain_community.document_loaders import JSONLoader
import json
from pathlib import Path
from pprint import pprint

def load_json_file(file_path):
    data = json.loads(Path(file_path).read_text()) # using json

    loader = JSONLoader(file_path=file_path, jq_schema='.[]',
    text_content=False)
    documents = loader.load()
    return documents

if __name__ == "__main__":
    file_path = "project_1_publications.json"
    documents = load_json_file(file_path)
    json_output = [doc.page_content for doc in documents[:2]]
    formatted = json.dumps(json_output, indent=4)
    pprint(formatted)