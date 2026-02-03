# master_dole.csv Data Quality Analysis Report

**Analysis Date:** 2026-01-31
**File Analyzed:** master_dole.csv (2,223 lines, 2,222 data rows)
**Total Issues Found:** 831+

---

## Executive Summary

Analysis of master_dole.csv revealed **8 major categories of data quality issues**, ranging from critical edition sequence errors to systematic missing edition numbers in recent data. The most significant findings include:

- **500+ entries (22%)** with "Unknown" edition numbers (2021-present)
- **3 critical backwards edition sequences**
- **146+ temporal gaps** exceeding 300 days
- **47 missing editions** in sequences
- **30+ duplicate dates** with conflicting data

---

## Critical Issues (Priority 1)

### 1. Backwards Edition Sequences

**Impact:** Data integrity violation - editions should increment, not decrement

| Location | Date Range | Edition Progression | Line # |
|----------|------------|---------------------|--------|
| Anchorage | 2016-11-01 → 2016-11-10 | Ed. 99 → Ed. 98 ⚠️ | 50 |
| Anchorage | 2020-10-30 → 2020-10-31 | Ed. 107 → Ed. 106 ⚠️ | 60 |
| Atlanta | 2019-02-28 → 2019-05-02 | Ed. 102 → Ed. 101 ⚠️ | 99 |

**Recommendation:** Verify source documents and correct edition numbers.

---

### 2. Conflicting Duplicate Dates

**Anchorage 2016-11-10** - THREE entries with inconsistent editions:
- Ed. 99 (NARA)
- Ed. 98 (NARA)
- Ed. 99 (NARA) - duplicate

**Atlanta 2015-08-20** - TWO entries with different editions:
- Ed. 92 (Web Archive)
- Ed. 95 (NARA)

**Recommendation:** Determine canonical edition numbers from authoritative sources.

---

## High Priority Issues (Priority 2)

### 3. Unknown Edition Numbers

**Scope:** 500+ entries (~22% of dataset)
**Date Range:** 2021-02-25 onwards
**Locations Affected:** All locations systematically

**Pattern:** All entries after 2021-02-25 have "Unknown" as the edition number, coinciding with shift from NARA archives to Web Archive PDF sources.

**Recommendation:** Extract edition numbers from PDF filenames or document metadata. Example pattern observed: filenames likely contain edition identifiers.

---

### 4. Large Temporal Gaps (>300 days)

**Total Count:** 146+ gaps
**Longest Gap:** Atlanta 2014-01-13 to 2015-08-20 = **584 days** (1.6 years)

#### Notable Gap Patterns:

**336-Day Gaps (Exactly 2× Standard Interval):**
- Multiple occurrences in 2022-2023 across locations
- Suggests systematic missing editions

**500+ Day Gaps (Early Data, 2011-2015):**
| Location | Start Date | End Date | Gap (days) |
|----------|------------|----------|------------|
| Atlanta | 2014-01-13 | 2015-08-20 | 584 |
| Anchorage | 2011-07-14 | 2012-12-20 | 524 |
| Albuquerque | 2011-11-03 | 2013-03-21 | 503 |

**Recommendation:** Search additional archives (FAA, NOAA, university libraries) for missing editions.

---

### 5. Missing Editions in Sequences

**Total Count:** 47 sequence gaps

| Location | Missing Editions | Notes |
|----------|------------------|-------|
| Albuquerque | 88, 89, 94 | Multiple gaps |
| Anchorage | 92, 95 | 2015-2016 period |
| Atlanta | 90, 93-94 | Ed. 93-94 missing despite duplicate date |
| Bethel | 56 | 2-year gap (2014-2016) |
| Billings | 85 | Single edition gap |

**Recommendation:** Investigate whether missing editions were unpublished or lost in archival process.

---

## Medium Priority Issues (Priority 3)

### 6. Interval Field Mismatches

**Total Count:** 23+ entries where `interval_days` field doesn't match actual date difference

**Worst Cases:**
| Location | Line | Stated Interval | Actual Gap | Error |
|----------|------|-----------------|------------|-------|
| Atlanta | 88 | 254 days | 584 days | +330 days |
| Anchorage | Various | Multiple mismatches | | |

**Recommendation:** Recalculate all `interval_days` fields programmatically to ensure accuracy.

---

### 7. Irregular Cadence

**Total Count:** 78+ entries with unexpected intervals

**Expected Patterns:**
- **56 days:** Standard bi-monthly cycle
- **168 days:** Semi-annual cycle (6 months)
- **196 days:** Extended cycle (7 months)
- **364 days:** Annual cycle (Bethel/Alaska charts)

**Irregular Intervals Found:**
- 531, 508, 584 days (>1 year gaps)
- 301, 259 days (between standard intervals)
- 1, 9, 10 days (rapid successive publications)

**Note:** Bethel's 364-day annual cycle is legitimate for Alaska region charts.

---

### 8. Duplicate Dates

**Total Count:** 30+ occurrences

**Pattern:** Most duplicates are from multiple data sources (NARA vs. Web Archive)

**Categories:**
1. **Same edition, different sources:** Legitimate redundancy (can deduplicate)
2. **Different editions, same date:** Data conflict (requires investigation)
3. **Triple entries:** Data quality issue (Anchorage 2016-11-10)

**Recommendation:** Implement deduplication strategy prioritizing NARA archives over Web Archive when editions match.

---

## Low Priority Issues (Priority 4)

### 9. Singleton Dates

**Total Count:** 4 entries
**Definition:** Isolated dates with >200 day gaps on both sides

These are typically first entries for locations in 2011-2012, representing initial data collection points rather than errors.

---

## Data Patterns Identified

### Three Distinct Data Source Eras:

1. **2011-2015: Web Archive Era (Poor Coverage)**
   - Large gaps (500+ days common)
   - Limited edition availability
   - Inconsistent coverage

2. **2016-2020: NARA Archives Era (Good Coverage)**
   - Regular intervals
   - Edition numbers known
   - Most reliable data

3. **2021-Present: Web Archive PDF Era (Good Temporal, Unknown Editions)**
   - Consistent publication dates
   - 500+ "Unknown" edition numbers
   - Needs edition extraction

### Publication Schedule Variations by Location:

- **Most charts:** 56, 168, or 196-day cycles
- **Bethel/Alaska charts:** 364-day annual cycles
- **Systematic 336-day gaps:** In 2022-2023 suggest coordinated missing editions

---

## Recommendations

### Priority 1 (Critical) - Immediate Action Required:
- [ ] Fix 3 backwards edition sequences
- [ ] Resolve conflicting duplicates (Atlanta 2015-08-20, Anchorage 2016-11-10)

### Priority 2 (High) - Address Within 1 Month:
- [ ] Extract edition numbers from 2021+ PDF filenames/metadata (500+ entries)
- [ ] Deduplicate entries, prioritizing NARA over Web Archive when appropriate
- [ ] Search for 47 missing editions in additional archives (FAA, NOAA, etc.)

### Priority 3 (Medium) - Address Within 3 Months:
- [ ] Recalculate all `interval_days` fields programmatically
- [ ] Document expected publication schedules per location
- [ ] Create validation script to flag irregular intervals

### Priority 4 (Low) - Future Enhancements:
- [ ] Add data provenance fields (`data_source`, `confidence_level`, `notes`)
- [ ] Create validation rules to prevent future errors
- [ ] Implement automated quality checks for new data additions

---

## Validation Rules for Future Data

Suggested rules to implement:

1. **Edition Numbers:** Must increment or stay same, never decrement
2. **Intervals:** Should be 56±7, 168±14, 196±14, or 364±14 days
3. **Duplicates:** Flag dates with conflicting edition numbers
4. **Gaps:** Flag intervals >300 days for review
5. **Unknown Editions:** Require explanation/source note

---

## Analysis Methodology

- **Tool:** Deep pattern analysis of CSV structure
- **Scope:** All 2,222 data rows analyzed
- **Focus Areas:** Date sequences, edition progressions, temporal gaps, data consistency
- **Validation:** Cross-referenced intervals, checked edition sequences, identified statistical outliers

---

*End of Report*
