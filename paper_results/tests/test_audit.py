import hashlib
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import audit


def verdict(winner):
    return {metric: {'Winner': winner, 'Explanation': 'Answer 1 and Answer 2 are discussed.'}
            for metric in audit.METRICS}


class AuditTests(unittest.TestCase):
    def test_swapped_passes_and_reversed_comparator(self):
        rows = [verdict(w) for w in ('Answer 1', 'Answer 2', 'Answer 2', 'Answer 2')]
        self.assertEqual(audit.summarize(rows, 2, 1)['Overall Winner']['wins'], 3)
        self.assertEqual(audit.summarize(rows, 2, 2)['Overall Winner']['wins'], 1)

    def test_does_not_parse_explanation_for_winner(self):
        result = audit.summarize([verdict('Answer 2'), verdict('Answer 1')], 1, 1)
        self.assertEqual(result['Overall Winner']['percent'], '0.00')

    def test_rejects_missing_extra_and_odd_row_counts(self):
        for rows in ([], [verdict('Answer 1')], [verdict('Answer 1')]*6):
            with self.assertRaises(ValueError):
                audit.summarize(rows, 2, 1)

    def test_rejects_invalid_and_missing_winners(self):
        for bad in ('Tie', None, 'answer 1'):
            with self.assertRaises(ValueError):
                audit.summarize([verdict(bad), verdict('Answer 1')], 1, 1)
        with self.assertRaises(ValueError):
            audit.summarize([{}, {}], 1, 1)

    def test_exact_rounding_and_unrounded_aggregate(self):
        self.assertEqual(audit.percent(146, 260), '56.15')
        values = [audit.score(w,n) for w,n in [(146,260),(110,200),(111,200),(108,200)]]
        self.assertEqual(audit.aggregate(values), {'macro': '55.16', 'pooled': '55.23'})

    def test_rejects_unsafe_paths(self):
        with tempfile.TemporaryDirectory() as tmp:
            for name in ('../secret', '/outside', 'missing.jsonl'):
                with self.assertRaises(ValueError):
                    audit.safe_path(Path(tmp), name)

    def test_validates_hash_and_json_without_rewriting(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp)/'a.jsonl'
            data = (json.dumps(verdict('Answer 1'))+'\n').encode()
            p.write_bytes(data)
            digest = hashlib.sha256(data).hexdigest()
            self.assertEqual(len(audit.read_jsonl(p,digest)),1)
            self.assertEqual(p.read_bytes(),data)
            with self.assertRaises(ValueError):
                audit.read_jsonl(p,'0'*64)

    def fixture_bundle(self, root):
        manifest, experiments = [], []
        for ds in audit.DATASETS:
            p = root/'raw'/f'{ds}.jsonl'
            p.parent.mkdir(exist_ok=True)
            p.write_text('\n'.join(json.dumps(verdict(w)) for w in ('Answer 1','Answer 2'))+'\n',encoding='utf8')
            manifest.append({'path':f'raw/{ds}.jsonl','role':'evaluation','dataset':ds,
                             'rows':2,'sha256':hashlib.sha256(p.read_bytes()).hexdigest()})
            experiments.append({'id':'comparison','dataset':ds,'evaluation_file':f'raw/{ds}.jsonl',
                                'target_first_pass_answer':1,'expected_percent':{'Overall Winner':100}})
        (root/'files.json').write_text(json.dumps({'files':manifest}),encoding='utf8')
        spec = {'experiments':experiments,'aggregate_expectations':{'comparison':{'macro':100,'pooled':100}}}
        (root/'experiments.json').write_text(json.dumps(spec),encoding='utf8')
        return spec

    def test_full_bundle_detects_paper_mismatch(self):
        with tempfile.TemporaryDirectory() as tmp, patch.dict(audit.DATASETS,dict.fromkeys(audit.DATASETS,1),clear=True):
            root = Path(tmp)
            spec = self.fixture_bundle(root)
            self.assertEqual(audit.verify(root)['comparisons_verified'],4)
            spec['experiments'][0]['expected_percent']['Overall Winner'] = 99
            (root/'experiments.json').write_text(json.dumps(spec),encoding='utf8')
            with self.assertRaisesRegex(ValueError,'Paper mismatch'):
                audit.verify(root)

    def test_full_bundle_detects_unlisted_raw_file(self):
        with tempfile.TemporaryDirectory() as tmp, patch.dict(audit.DATASETS,dict.fromkeys(audit.DATASETS,1),clear=True):
            root = Path(tmp)
            self.fixture_bundle(root)
            (root/'raw/unlisted.jsonl').write_text('{}\n',encoding='utf8')
            with self.assertRaisesRegex(ValueError,'manifest do not match'):
                audit.verify(root)


if __name__ == '__main__':
    unittest.main()
