# Corpus Extraction

We use the JSONL dump from the British Library website. Two resources needs to be downloaded (and extracted):

* [Digitised Books. c. 1510 - c. 1900. JSON (OCR derived text)](https://data.bl.uk/digbks/db14.html)
* [Digitised 19th Century Books - Metadata - 01/09/2013](https://data.bl.uk/digbks/DB21.html)

After extracting, the corpus can be extracted via the following script:

```python
import json

from collections import defaultdict
from langdetect import detect
from pathlib import Path
from tqdm.notebook import tqdm

root_dir = Path("./json")
file_list = list(root_dir.rglob("*_text.json"))
print(f"Found {len(file_list)} JSON files")

# Metadata stuff
with open("./book_data.json", "rt") as f_p:
    metadata = json.load(f_p)

identifier_year_mapping = {}

for metadata_entry in metadata:
    identifier = metadata_entry["identifier"]
    year = metadata_entry["date"]
    identifier_year_mapping[identifier] = year

# Actual extracting
output_filename = "./bl_1800-1900_extracted.txt"
output_fp = open(output_filename, "wt")

for filename in tqdm(file_list):
    with open(filename, "rt") as f_p:
        pages = json.load(f_p)
    book_text = "\n".join(page[1] for page in pages if page[1].strip() != "")
    
    if not book_text:
        continue
    try:
        detected_language = detect(book_text[:1000])
    except:
        print(f"Skipping {filename.name}...")
        continue
    
    if detected_language != "en":
        continue
    
    current_identifier = filename.name.split("_")[0]
    current_year = identifier_year_mapping.get(current_identifier)
    
    if not current_year:
        continue
    
    current_year = int(current_year)
    
    if current_year >= 1800 and current_year < 1900:
        output_fp.write(book_text + "\n")
```

It shows that 63,985 JSON files were found. Additionally, language detection using `langdetect` is performed to only extract English texts. Metadata information is used to extract for a certain time period (>=1800 and <1900).
