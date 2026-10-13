import importlib.util
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

spec = importlib.util.spec_from_file_location('publisher', Path(__file__).resolve().parents[1] / 'scripts/publish_chart_pmtiles_from_reslice.py')
publisher = importlib.util.module_from_spec(spec)
spec.loader.exec_module(publisher)


class PublicationTests(unittest.TestCase):
    record = {'remote': 'r2:hashed', 'path': 'chart/a.hash', 'size': 42, 'sha256': 'abc'}

    def test_missing_or_wrong_hash_never_produces_receipt(self):
        for sha in (None, 'wrong'):
            with patch.object(publisher, 'stat_object', return_value={'size': 42, 'sha256': sha}):
                with self.assertRaisesRegex(RuntimeError, 'sha256'):
                    publisher.publish_one('chart/a', self.record, 'r2:charts', 'sectionals', 'run', False)

    def test_refresh_forces_replacement_and_preserves_metadata(self):
        with patch.object(publisher.subprocess, 'run', return_value=SimpleNamespace(returncode=0)) as run, patch.object(publisher, 'stat_object', return_value={'size': 42, 'sha256': 'abc'}):
            receipt = publisher.publish_one('chart/a', self.record, 'r2:charts', 'sectionals', 'run', True)
        self.assertIn('--ignore-times', run.call_args.args[0])
        self.assertIn('--metadata', run.call_args.args[0])
        self.assertEqual(receipt['sha256'], 'abc')

    def test_success_survives_later_failure(self):
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / 'receipts.jsonl'
            args = SimpleNamespace(uploads=output, workers=1, remote='r2:charts', prefix='sectionals', run=Path('run'), copy=True)
            def publish(key, *unused):
                if key == 'bad':
                    raise RuntimeError('copy failed')
                return {'key': key, 'size': 42, 'sha256': 'abc'}
            def completed(futures):
                by_key = {key: future for future, key in futures.items()}
                yield by_key['good']
                # Inspect while the batch is still active, before its failure.
                self.assertEqual(json.loads(output.read_text())['key'], 'good')
                yield by_key['bad']
            with patch.object(publisher, 'publish_one', side_effect=publish), patch.object(publisher, 'as_completed', side_effect=completed):
                ok, bad = publisher.publish_ready(['good', 'bad'], {'good': self.record, 'bad': self.record}, args)
            self.assertEqual((ok, bad), (1, 1))
            self.assertEqual(json.loads(output.read_text())['key'], 'good')


if __name__ == '__main__':
    unittest.main()
