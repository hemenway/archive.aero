#!/usr/bin/env python3
"""
Analyze master_dole.csv for errors and visualize missing editions as timeline gaps
"""

import csv
import os
import argparse
from datetime import datetime
from collections import defaultdict

def parse_date(date_str):
    """Parse date string to datetime object"""
    if not date_str:
        return None
    try:
        return datetime.strptime(date_str, '%Y-%m-%d')
    except ValueError:
        return None

def analyze_csv(filepath):
    """Analyze the CSV file for errors and structure"""

    errors = []
    locations_data = defaultdict(list)

    with open(filepath, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        line_num = 2  # Start at 2 because of header

        for row in reader:
            download_link = row.get('download_link', '')
            date_str = row.get('date', '')
            location = row.get('location', '')
            edition = row.get('edition', '')
            end_date_str = row.get('end_date', '')

            # Check for missing required fields
            if not download_link:
                errors.append(f"Line {line_num}: Missing download_link")
            if not date_str:
                errors.append(f"Line {line_num}: Missing date")
            if not location:
                errors.append(f"Line {line_num}: Missing location")
            if not edition:
                errors.append(f"Line {line_num}: Missing edition")

            # Validate date format
            date_obj = parse_date(date_str)
            if date_str and not date_obj:
                errors.append(f"Line {line_num}: Invalid date format '{date_str}'")

            end_date_obj = parse_date(end_date_str)
            if end_date_str and not end_date_obj:
                errors.append(f"Line {line_num}: Invalid end_date format '{end_date_str}'")

            # Check if end_date is after date
            if date_obj and end_date_obj and end_date_obj <= date_obj:
                errors.append(f"Line {line_num}: end_date ({end_date_str}) is not after date ({date_str})")

            # Validate interval_days is numeric when present
            interval_days = row.get('interval_days', '')
            if interval_days:
                try:
                    int(interval_days)
                except ValueError:
                    errors.append(f"Line {line_num}: Invalid interval_days '{interval_days}'")

            # Store data by location
            if location:
                locations_data[location].append({
                    'line': line_num,
                    'date': date_obj,
                    'end_date': end_date_obj,
                    'edition': edition,
                    'download_link': download_link,
                    'tif_filename': row.get('tif_filename', '')
                })

            line_num += 1

    return errors, locations_data

def find_missing_editions(locations_data):
    """Find missing edition numbers for each location"""

    missing_by_location = {}

    for location, records in locations_data.items():
        # Filter out "Unknown" editions and extract numeric editions
        numeric_editions = []
        for record in records:
            if record['edition'] != 'Unknown':
                try:
                    numeric_editions.append(int(record['edition']))
                except ValueError:
                    pass  # Skip non-numeric editions

        if not numeric_editions:
            continue

        numeric_editions = sorted(set(numeric_editions))

        # Find gaps in the sequence
        if len(numeric_editions) > 1:
            min_ed = min(numeric_editions)
            max_ed = max(numeric_editions)
            full_range = set(range(min_ed, max_ed + 1))
            missing = sorted(full_range - set(numeric_editions))

            if missing:
                missing_by_location[location] = {
                    'range': f"{min_ed}-{max_ed}",
                    'found': numeric_editions,
                    'missing': missing,
                    'count_found': len(numeric_editions),
                    'count_missing': len(missing)
                }

    return missing_by_location

def find_date_gaps(locations_data):
    """Find significant gaps in publication dates for each location"""

    gaps_by_location = {}

    for location, records in locations_data.items():
        # Sort records by date
        sorted_records = sorted([r for r in records if r['date']],
                               key=lambda x: x['date'])

        if len(sorted_records) < 2:
            continue

        gaps = []
        for i in range(len(sorted_records) - 1):
            current = sorted_records[i]
            next_rec = sorted_records[i + 1]

            # Calculate expected next date from end_date
            if current['end_date']:
                expected = current['end_date']
                actual = next_rec['date']
                gap_days = (actual - expected).days

                # Report gaps of more than 60 days
                if gap_days > 60:
                    gaps.append({
                        'after_date': current['date'].strftime('%Y-%m-%d'),
                        'after_edition': current['edition'],
                        'expected': expected.strftime('%Y-%m-%d'),
                        'next_date': actual.strftime('%Y-%m-%d'),
                        'next_edition': next_rec['edition'],
                        'gap_days': gap_days
                    })

        if gaps:
            gaps_by_location[location] = gaps

    return gaps_by_location

def build_file_index(tif_dir='/Volumes/drive/newrawtiffs'):
    """Build an index of files in the tif directory for preview linking"""
    import os
    file_index = {}
    if not os.path.isdir(tif_dir):
        return file_index
    for fname in os.listdir(tif_dir):
        file_index[fname] = fname
        # Also index without extension for fallback matching
        base, ext = os.path.splitext(fname)
        if base not in file_index or ext == '.tif':
            file_index[base] = fname
    return file_index

def resolve_tif_file(tif_filename, file_index):
    """Find the best previewable file for a given tif_filename"""
    if not tif_filename:
        return ''
    # Direct match
    if tif_filename in file_index:
        candidate = file_index[tif_filename]
        if candidate.endswith('.tif'):
            return candidate
    # Try without .zip extension
    import os
    base, ext = os.path.splitext(tif_filename)
    if ext == '.zip' and base in file_index:
        candidate = file_index[base]
        if candidate.endswith('.tif'):
            return candidate
        # The extensionless file might be a tif
        return candidate
    # Try adding .tif
    tif_name = base + '.tif'
    if tif_name in file_index:
        return file_index[tif_name]
    # Return original if it exists at all
    if tif_filename in file_index:
        return file_index[tif_filename]
    return tif_filename

def create_html_visualization(locations_data, missing_editions, date_gaps, output_file='timeline_gaps.html'):
    """Create an HTML visualization showing coverage gaps"""

    # Build file index for preview links
    file_index = build_file_index()

    # Prepare data for visualization
    all_dates = []
    for records in locations_data.values():
        for r in records:
            if r['date']:
                all_dates.append(r['date'])

    if not all_dates:
        print("No dates found for visualization")
        return

    min_date = min(all_dates)
    max_date = max(all_dates)

    # Show every location in case-insensitive alphabetical order
    sorted_locations = sorted(locations_data.keys(), key=lambda name: (name.lower(), name))

    html_content = f"""<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <title>Sectional Chart Coverage Timeline</title>
    <style>
        body {{
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            margin: 20px;
            background-color: #f5f5f5;
        }}
        h1 {{
            color: #333;
            text-align: center;
        }}
        .summary {{
            background: white;
            padding: 20px;
            margin: 20px 0;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }}
        .timeline {{
            background: white;
            padding: 20px;
            margin: 20px 0;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            overflow-x: auto;
        }}
        .location-row {{
            display: flex;
            align-items: center;
            margin: 5px 0;
            height: 30px;
        }}
        .location-label {{
            width: 150px;
            font-size: 12px;
            font-weight: 500;
            flex-shrink: 0;
        }}
        .timeline-track {{
            flex-grow: 1;
            height: 20px;
            position: relative;
            background: #f0f0f0;
            border-radius: 3px;
        }}
        .timeline-segment {{
            position: absolute;
            height: 18px;
            top: 1px;
            border-radius: 2px;
            border: 1px solid rgba(0,0,0,0.2);
            cursor: pointer;
            transition: opacity 0.15s;
        }}
        .timeline-segment:hover {{
            opacity: 0.7;
        }}
        .known-edition {{
            background-color: #4682b4;
        }}
        .unknown-edition {{
            background-color: #ff8c00;
        }}
        .legend {{
            margin: 20px 0;
            display: flex;
            gap: 20px;
            justify-content: center;
        }}
        .legend-item {{
            display: flex;
            align-items: center;
            gap: 8px;
        }}
        .legend-box {{
            width: 30px;
            height: 18px;
            border-radius: 3px;
            border: 1px solid rgba(0,0,0,0.2);
        }}
        .missing-editions {{
            background: white;
            padding: 20px;
            margin: 20px 0;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }}
        .location-group {{
            margin: 15px 0;
            padding: 10px;
            background: #f9f9f9;
            border-left: 4px solid #4682b4;
        }}
        table {{
            width: 100%;
            border-collapse: collapse;
            margin-top: 10px;
        }}
        th, td {{
            padding: 8px;
            text-align: left;
            border-bottom: 1px solid #ddd;
        }}
        th {{
            background-color: #4682b4;
            color: white;
        }}
        .error {{
            color: #d32f2f;
        }}
        .warning {{
            color: #f57c00;
        }}
        .preview-overlay {{
            display: none;
            position: fixed;
            top: 0; left: 0; right: 0; bottom: 0;
            background: rgba(0,0,0,0.8);
            z-index: 1000;
            justify-content: center;
            align-items: center;
            flex-direction: column;
        }}
        .preview-overlay.active {{
            display: flex;
        }}
        .preview-info {{
            color: white;
            font-size: 14px;
            margin-bottom: 10px;
            text-align: center;
            max-width: 90vw;
            word-break: break-all;
        }}
        .preview-info h3 {{
            margin: 0 0 5px 0;
        }}
        .preview-img {{
            max-width: 90vw;
            max-height: 80vh;
            object-fit: contain;
            background: white;
            border-radius: 4px;
        }}
        .preview-close {{
            position: fixed;
            top: 15px; right: 25px;
            color: white;
            font-size: 36px;
            cursor: pointer;
            z-index: 1001;
            line-height: 1;
        }}
        .preview-close:hover {{
            color: #ccc;
        }}
        .preview-open-link {{
            color: #7eb8ff;
            margin-top: 10px;
            font-size: 13px;
        }}
    </style>
</head>
<body>
    <h1>Sectional Chart Coverage Timeline Analysis</h1>

    <div class="summary">
        <h2>Summary Statistics</h2>
        <p><strong>Total Locations:</strong> {len(locations_data)}</p>
        <p><strong>Total Records:</strong> {sum(len(records) for records in locations_data.values())}</p>
        <p><strong>Date Range:</strong> {min_date.strftime('%Y-%m-%d')} to {max_date.strftime('%Y-%m-%d')}</p>
        <p><strong>Locations with Missing Editions:</strong> {len(missing_editions)}</p>
        <p><strong>Locations with Date Gaps:</strong> {len(date_gaps)}</p>
    </div>

    <div class="legend">
        <div class="legend-item">
            <div class="legend-box known-edition"></div>
            <span>Known Edition</span>
        </div>
        <div class="legend-item">
            <div class="legend-box unknown-edition"></div>
            <span>Unknown Edition</span>
        </div>
    </div>
"""

    # Add timeline visualization
    html_content += (
        f'    <div class="timeline">\n'
        f'        <h2>Coverage Timeline (All {len(sorted_locations)} Locations, Alphabetical)</h2>\n'
    )

    total_days = (max_date - min_date).days

    for location in sorted_locations:
        records = locations_data[location]
        sorted_records = sorted([r for r in records if r['date'] and r['end_date']],
                               key=lambda x: x['date'])

        if not sorted_records:
            continue

        html_content += f'        <div class="location-row">\n'
        html_content += f'            <div class="location-label">{location}</div>\n'
        html_content += f'            <div class="timeline-track">\n'

        for record in sorted_records:
            start_offset = (record['date'] - min_date).days
            duration = (record['end_date'] - record['date']).days

            left_pct = (start_offset / total_days) * 100
            width_pct = (duration / total_days) * 100

            edition_class = 'known-edition' if record['edition'] != 'Unknown' else 'unknown-edition'
            title = f"{record['edition']} ({record['date'].strftime('%Y-%m-%d')} to {record['end_date'].strftime('%Y-%m-%d')})"

            tif_file = resolve_tif_file(record.get('tif_filename', ''), file_index)
            html_content += f'                <div class="timeline-segment {edition_class}" '
            html_content += f'style="left: {left_pct:.2f}%; width: {width_pct:.2f}%;" '
            html_content += f'title="{title}" '
            html_content += f'data-file="{tif_file}" '
            html_content += f'data-location="{location}" '
            html_content += f'data-edition="{record["edition"]}" '
            html_content += f'onclick="showPreview(this)"></div>\n'

        html_content += '            </div>\n'
        html_content += '        </div>\n'

    html_content += '    </div>\n'

    # Add missing editions section
    if missing_editions:
        html_content += '    <div class="missing-editions">\n'
        html_content += '        <h2>Missing Edition Numbers</h2>\n'

        for location in sorted(missing_editions.keys()):
            info = missing_editions[location]
            html_content += f'        <div class="location-group">\n'
            html_content += f'            <h3>{location}</h3>\n'
            html_content += f'            <p><strong>Edition Range:</strong> {info["range"]}</p>\n'
            html_content += f'            <p><strong>Found:</strong> {info["count_found"]} editions</p>\n'
            html_content += f'            <p class="warning"><strong>Missing:</strong> {info["count_missing"]} editions: '
            html_content += ', '.join(map(str, info['missing'][:50]))
            if len(info['missing']) > 50:
                html_content += f' ... and {len(info["missing"]) - 50} more'
            html_content += '</p>\n'
            html_content += '        </div>\n'

        html_content += '    </div>\n'

    # Add date gaps section
    if date_gaps:
        html_content += '    <div class="missing-editions">\n'
        html_content += '        <h2>Significant Date Gaps (>60 days)</h2>\n'

        for location in sorted(date_gaps.keys()):
            gaps = date_gaps[location]
            html_content += f'        <div class="location-group">\n'
            html_content += f'            <h3>{location}</h3>\n'
            html_content += '            <table>\n'
            html_content += '                <tr><th>After Edition</th><th>Expected Next</th><th>Actual Next</th><th>Gap (days)</th></tr>\n'

            for gap in gaps:
                html_content += f'                <tr>\n'
                html_content += f'                    <td>{gap["after_edition"]} ({gap["after_date"]})</td>\n'
                html_content += f'                    <td>{gap["expected"]}</td>\n'
                html_content += f'                    <td class="warning">{gap["next_date"]} (Ed. {gap["next_edition"]})</td>\n'
                html_content += f'                    <td class="error">{gap["gap_days"]}</td>\n'
                html_content += '                </tr>\n'

            html_content += '            </table>\n'
            html_content += '        </div>\n'

        html_content += '    </div>\n'

    html_content += """
    <div class="preview-overlay" id="previewOverlay" onclick="closePreview(event)">
        <span class="preview-close" onclick="closePreview(event)">&times;</span>
        <div class="preview-info">
            <h3 id="previewTitle"></h3>
            <p id="previewFilename"></p>
        </div>
        <div id="previewLoading" style="display:none; color:white; font-size:18px;">Converting image...</div>
        <img class="preview-img" id="previewImg" src="" alt="Preview" />
    </div>
    <script>
        function showPreview(el) {
            const file = el.dataset.file;
            const loc = el.dataset.location;
            const edition = el.dataset.edition;

            if (!file) {
                alert('No file associated with this segment');
                return;
            }

            document.getElementById('previewTitle').textContent = loc + ' - Edition ' + edition;
            document.getElementById('previewFilename').textContent = file;
            document.getElementById('previewImg').style.display = 'none';
            document.getElementById('previewLoading').style.display = 'block';
            document.getElementById('previewOverlay').classList.add('active');

            const img = document.getElementById('previewImg');
            img.onload = function() {
                document.getElementById('previewLoading').style.display = 'none';
                img.style.display = 'block';
            };
            img.onerror = function() {
                document.getElementById('previewLoading').textContent = 'Failed to load preview for: ' + file;
            };
            img.src = '/preview/' + encodeURIComponent(file);
        }

        function closePreview(e) {
            if (e.target === document.getElementById('previewOverlay') ||
                e.target.classList.contains('preview-close')) {
                document.getElementById('previewOverlay').classList.remove('active');
                document.getElementById('previewImg').src = '';
                document.getElementById('previewLoading').style.display = 'none';
            }
        }

        document.addEventListener('keydown', function(e) {
            if (e.key === 'Escape') {
                document.getElementById('previewOverlay').classList.remove('active');
                document.getElementById('previewImg').src = '';
                document.getElementById('previewLoading').style.display = 'none';
            }
        });
    </script>
</body>
</html>
"""

    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(html_content)

    print(f"\n   HTML visualization saved to: {output_file}")
    return output_file

def main():
    parser = argparse.ArgumentParser(
        description="Analyze master_dole-style CSV and generate timeline gap visualization"
    )
    parser.add_argument(
        "--csv",
        default='/Users/ryanhemenway/archive.aero/master_dole.csv',
        help="Input CSV path (default: master_dole.csv in this workspace)",
    )
    parser.add_argument(
        "--output-html",
        default="timeline_gaps.html",
        help="Output HTML path (default: timeline_gaps.html)",
    )
    args = parser.parse_args()

    csv_file = args.csv

    print("="*80)
    print("MASTER_DOLE.CSV ANALYSIS")
    print("="*80)
    print(f"\nInput CSV: {csv_file}")

    # Analyze CSV
    print("\n1. Checking for errors...")
    errors, locations_data = analyze_csv(csv_file)

    if errors:
        print(f"\n   Found {len(errors)} error(s):")
        for error in errors[:50]:  # Show first 50 errors
            print(f"   - {error}")
        if len(errors) > 50:
            print(f"   ... and {len(errors) - 50} more errors")
    else:
        print("   ✓ No errors found!")

    # Find missing editions
    print("\n2. Finding missing edition numbers...")
    missing_editions = find_missing_editions(locations_data)

    if missing_editions:
        print(f"\n   Found gaps in edition sequences for {len(missing_editions)} location(s):")
        print(f"   (Top 10 by number of missing editions):\n")

        sorted_missing = sorted(missing_editions.items(),
                              key=lambda x: x[1]['count_missing'],
                              reverse=True)

        for location, info in sorted_missing[:10]:
            print(f"   {location}:")
            print(f"      Range: {info['range']}")
            print(f"      Found: {info['count_found']} | Missing: {info['count_missing']}")
            print(f"      Missing editions: {info['missing'][:15]}")
            if len(info['missing']) > 15:
                print(f"      ... and {len(info['missing']) - 15} more")
            print()
    else:
        print("   ✓ No missing editions found!")

    # Find date gaps
    print("\n3. Finding significant date gaps...")
    date_gaps = find_date_gaps(locations_data)

    if date_gaps:
        total_gaps = sum(len(gaps) for gaps in date_gaps.values())
        print(f"\n   Found {total_gaps} significant gap(s) across {len(date_gaps)} location(s)")
    else:
        print("   ✓ No significant date gaps found!")

    # Summary statistics
    print("\n4. Summary Statistics:")
    print(f"   Total locations: {len(locations_data)}")
    print(f"   Total records: {sum(len(records) for records in locations_data.values())}")

    unknown_count = sum(1 for records in locations_data.values()
                       for r in records if r['edition'] == 'Unknown')
    print(f"   Records with 'Unknown' edition: {unknown_count}")

    # Find date range
    all_dates = [r['date'] for records in locations_data.values()
                 for r in records if r['date']]
    if all_dates:
        print(f"   Date range: {min(all_dates).strftime('%Y-%m-%d')} to {max(all_dates).strftime('%Y-%m-%d')}")

    # Create visualization
    print("\n5. Creating HTML visualization...")
    try:
        create_html_visualization(
            locations_data,
            missing_editions,
            date_gaps,
            output_file=args.output_html,
        )
        print("   ✓ Done!")
    except Exception as e:
        print(f"   Error creating visualization: {e}")
        import traceback
        traceback.print_exc()

    print("\n" + "="*80)

if __name__ == '__main__':
    main()
