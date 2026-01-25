#!/bin/bash
# CDX API Examples for verifying and filtering FAA chart archives

# 1. Get only unique files (by content hash) - eliminates duplicate captures
curl "https://web.archive.org/cdx/search/cdx?\
url=aeronav.faa.gov/visual/*.pdf&\
output=json&\
collapse=digest&\
filter=statuscode:200"

# 2. Get all captures of a specific file with size verification
curl "https://web.archive.org/cdx/search/cdx?\
url=aeronav.faa.gov/visual/02-20-2025/PDFs/Washington.pdf&\
output=json&\
fl=timestamp,original,length,digest"

# 3. Get only PDFs larger than 100MB (likely full charts, not thumbnails)
curl "https://web.archive.org/cdx/search/cdx?\
url=aeronav.faa.gov/visual/*.pdf&\
output=json&\
filter=statuscode:200&\
filter=mimetype:application/pdf" | \
jq -r '.[] | select(.[6] | tonumber > 100000000)'

# 4. Check if a file changed over time (different digests)
curl "https://web.archive.org/cdx/search/cdx?\
url=aeronav.faa.gov/visual/*/PDFs/Washington.pdf&\
output=json&\
collapse=digest" | jq 'length - 1'  # Number of unique versions

# 5. Get date range (only 2024 captures)
curl "https://web.archive.org/cdx/search/cdx?\
url=aeronav.faa.gov/visual/*.pdf&\
from=20240101&\
to=20241231&\
output=json"

# 6. Verify file completeness before download
# Returns: timestamp, url, filesize, hash
curl "https://web.archive.org/cdx/search/cdx?\
url=aeronav.faa.gov/visual/02-20-2025/PDFs/Washington.pdf&\
output=json&\
fl=timestamp,original,length,digest&\
filter=statuscode:200" | jq -r '.[]'

# 7. Pagination for huge result sets (use resumeKey)
curl "https://web.archive.org/cdx/search/cdx?\
url=aeronav.faa.gov/visual/*&\
output=json&\
showResumeKey=true&\
limit=1000"

# Then use the resumeKey from the last row to continue:
# &resumeKey=<value_from_last_result>
