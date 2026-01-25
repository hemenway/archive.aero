#!/usr/bin/env python3
import urllib.request
import json
import csv
from collections import defaultdict
from urllib.parse import urlencode

# Query CDX API for all captures
base_url = "https://web.archive.org/cdx/search/cdx"
params = {
    "url": "aeronav.faa.gov/visual/*",
    "output": "json",
    "filter": "statuscode:200",  # Only successful captures
    "fl": "timestamp,original,mimetype,statuscode,digest,length",  # Select specific fields
}

query_string = urlencode(params)
full_url = f"{base_url}?{query_string}"

print("Fetching data from Wayback Machine CDX API...")
with urllib.request.urlopen(full_url) as response:
    data = json.loads(response.read().decode())

# Parse results
# Format: [timestamp, original_url, statuscode, mimetype, length, redirects, robotflags]
if not data or len(data) < 2:
    print(f"Error: Unexpected response format. Got: {data}")
    exit(1)

headers = data[0]
results = data[1:]

print(f"Found {len(results)} total captures\n")

# Group by URL to find first and last capture
# Column indices: 0=timestamp, 1=original, 2=mimetype, 3=statuscode, 4=digest, 5=length
url_data = defaultdict(lambda: {
    "first": None, "last": None, "mime": None,
    "digest": None, "length": None, "captures": 0
})

for row in results:
    timestamp = row[0]
    original_url = row[1]
    mimetype = row[2] if len(row) > 2 else "unknown"
    digest = row[4] if len(row) > 4 else ""
    length = row[5] if len(row) > 5 else "0"

    if original_url not in url_data:
        url_data[original_url]["first"] = timestamp
        url_data[original_url]["mime"] = mimetype
        url_data[original_url]["digest"] = digest
        url_data[original_url]["length"] = length

    url_data[original_url]["last"] = timestamp
    url_data[original_url]["captures"] += 1

# Write to CSV
output_file = "wayback_faa_visual.csv"
with open(output_file, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow([
        "URL", "MIME Type", "First Capture", "Last Capture",
        "Total Captures", "File Size (bytes)", "Content Digest (SHA-1)"
    ])

    for original_url in sorted(url_data.keys()):
        url_info = url_data[original_url]
        writer.writerow([
            original_url,
            url_info["mime"],
            url_info["first"],
            url_info["last"],
            url_info["captures"],
            url_info["length"],
            url_info["digest"]
        ])

print(f"Saved to {output_file}")
print(f"Total unique URLs: {len(url_data)}")
