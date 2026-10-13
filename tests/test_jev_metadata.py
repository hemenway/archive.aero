"""Cached-output regressions found during worklist 16 acceptance review."""
import sys
import unittest
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import airfields_freeman_dates_jev as dates
import atc_posts_jev as atc


def airfield(start="", end="1934", start_basis="", end_basis="closed_stated"):
    return {"name": "Test", "state": "AR", "url": "https://example.org/test",
            "start_year": start, "end_year": end, "start_basis": start_basis,
            "end_basis": end_basis, "status": "gone", "last_known_year": ""}


def judgment(start=1939, end=1945):
    return {"jev_start_year": start, "jev_start_basis": "jev_built", "jev_start_conf": .99,
            "jev_end_year": end, "jev_end_basis": "jev_closed", "jev_end_conf": .99,
            "jev_still_open": 0.0}


class FreemanMergeTests(unittest.TestCase):
    def test_start_fill_cannot_cross_retained_end(self):
        row, review = airfield(), []
        dates.merge(row, judgment(), review)
        self.assertEqual(row["start_year"], "")
        self.assertEqual(row["end_year"], "1934")
        self.assertIn("rejected_date_conflict", {r["disposition"] for r in review})

    def test_weak_start_override_cannot_cross_retained_end(self):
        row = airfield("1930", start_basis="after_absence")
        review = []
        dates.merge(row, judgment(), review)
        self.assertEqual(row["start_year"], "1930")
        self.assertEqual([r["disposition"] for r in review if r["field"] == "start"],
                         ["rejected_date_conflict"])

    def test_start_and_weak_end_can_advance_together(self):
        for start, basis in (("", ""), ("1930", "after_absence")):
            with self.subTest(start=start):
                row = airfield(start, start_basis=basis, end_basis="last_seen")
                review = []
                actions = dates.merge(row, judgment(), review)
                self.assertEqual((row["start_year"], row["end_year"]), (1939, 1945))
                self.assertNotIn("start=jev_conflict", actions)
                self.assertNotIn("rejected_date_conflict", {r["disposition"] for r in review})

    def test_end_override_cannot_cross_retained_start(self):
        for basis in ("last_seen", "closed_stated"):
            with self.subTest(basis=basis):
                row = airfield("1966", "2002", "earliest", basis)
                dates.merge(row, judgment(1940, 1947), [],
                            "The field was depicted as abandoned on the 2002 chart.")
                self.assertEqual(row["end_year"], "2002")

    def test_consistent_conflict_resolution_still_replaces_both(self):
        row = airfield("1950", "1934", "earliest", "conflict")
        dates.merge(row, judgment(), [])
        self.assertEqual((row["start_year"], row["end_year"]), (1939, 1945))

    def test_future_year_cannot_be_adopted(self):
        row = airfield("1930", "", "built", "")
        dates.merge(row, judgment(dates.THIS_YEAR + 1, dates.THIS_YEAR + 2), [])
        self.assertEqual((row["start_year"], row["end_year"]), ("1930", ""))

    def test_em_dash_closure_interval(self):
        self.assertTrue(dates.range_pair("Closed between 2003—2015.", 2003, 2015))


class AtcMetadataTests(unittest.TestCase):
    def test_numeric_class_dates_and_curly_decades_are_candidates(self):
        post = {"title": "Class", "text": "Graduated 02/18/83; 7/11/01; 1980’s"}
        *_, years = atc.candidates(post, None)
        self.assertTrue({"1983", "2001", "1980s"}.issubset(years))
        self.assertEqual(atc.state_abbr("mo"), "MO")

    def test_historical_ident_does_not_override_stated_state(self):
        class FakeGazetteer:
            fss_name = {}

            def by_ident(self, ident, state=None):
                if state == "MO":
                    return None, ""
                raise AssertionError("must not retry KCQ without Missouri constraint")

            def by_city(self, city, state):
                self.city = (city, state)
                return {"lat": 38.75, "lon": -90.37, "name": "St Louis"}, "city1988"

        def chosen(value):
            return {"choice": value, "confidence": 1.0}

        answers = {"page_kind": chosen("roster_or_list"), "facility_type": chosen("fss"),
                   "facility_ident": chosen("KCQ"), "place": chosen("St. Louis, Missouri"),
                   "about_one_facility": {"noul": 1.0}, "about_a_person": {"noul": 0.0}}
        post = {"href": "/atc/st-louis", "title": "St. Louis", "categories": [],
                "image": "", "pdf_links": []}
        gaz = FakeGazetteer()
        meta = atc.interpret(post, answers, gaz, None)
        self.assertEqual(gaz.city, ("St. Louis", "MO"))
        self.assertEqual(meta["coord_source"], "city1988")

    def test_city_join_does_not_use_hospital_heliport(self):
        gaz = atc.Gazetteer.__new__(atc.Gazetteer)
        gaz.city1988 = defaultdict(list)
        gaz.oa_city = {("chesterfield", "MO"): [{"type": "heliport", "rank": 5}]}
        self.assertEqual(gaz.by_city("Chesterfield", "MO"), (None, ""))

    def test_public_record_retains_uncertainty_and_raw_year(self):
        class Meta(dict):
            def __missing__(self, key):
                return ""
        result = atc.public_record(Meta(year=1980, year_raw="1980s", year_conf=.8,
                                        opened="1980", opened_conf=.08))
        self.assertEqual(result["year_raw"], "1980s")
        self.assertEqual(result["opened_conf"], .08)
        self.assertIn("about_one_facility", result)


if __name__ == "__main__":
    unittest.main()
