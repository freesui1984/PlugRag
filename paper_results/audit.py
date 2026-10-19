"""Offline, read-only verification of paper result files and win-rate tables."""
from __future__ import annotations

import argparse
import hashlib
import json
from collections import defaultdict
from decimal import Decimal, ROUND_HALF_UP
from pathlib import Path

METRICS = ('Comprehensiveness', 'Empowerment', 'Diversity', 'Overall Winner')
DATASETS = {'mix': 130, 'cs': 100, 'legal': 100, 'agriculture': 100}


def percent(wins, total):
    if total <= 0 or not 0 <= wins <= total:
        raise ValueError('Invalid win count or denominator')
    return str((Decimal(wins)*100/Decimal(total)).quantize(Decimal('0.01'), rounding=ROUND_HALF_UP))


def score(wins, total):
    return {'wins': wins, 'judgments': total, 'percent': percent(wins,total)}


def summarize(rows, questions, target_first_pass_answer):
    if questions <= 0 or len(rows) != questions*2:
        raise ValueError(f'Expected {questions*2} judgment rows, got {len(rows)}')
    if target_first_pass_answer not in (1,2):
        raise ValueError('Comparator orientation must be 1 or 2')
    totals = dict.fromkeys(METRICS,0)
    for i,row in enumerate(rows):
        target = target_first_pass_answer if i < questions else 3-target_first_pass_answer
        for metric in METRICS:
            value = row.get(metric) if isinstance(row,dict) else None
            winner = value.get('Winner') if isinstance(value,dict) else None
            if winner not in ('Answer 1','Answer 2'):
                raise ValueError(f'Row {i+1}: invalid {metric} Winner')
            totals[metric] += winner == f'Answer {target}'
    return {k: score(v,len(rows)) for k,v in totals.items()}


def aggregate(values):
    if not values:
        raise ValueError('Empty aggregate')
    macro = sum(Decimal(v['wins'])*100/Decimal(v['judgments']) for v in values)/len(values)
    return {'macro': str(macro.quantize(Decimal('0.01'),rounding=ROUND_HALF_UP)),
            'pooled': percent(sum(v['wins'] for v in values),sum(v['judgments'] for v in values))}


def safe_path(root, relative):
    root = root.resolve()
    path = (root/relative).resolve()
    if not path.is_relative_to(root) or not path.is_file():
        raise ValueError(f'Unsafe or missing artifact: {relative}')
    return path


def read_jsonl(path, sha256):
    raw = path.read_bytes()
    if hashlib.sha256(raw).hexdigest() != sha256:
        raise ValueError(f'Checksum mismatch: {path.name}')
    rows = []
    for i,line in enumerate(raw.decode('utf-8-sig').splitlines(),1):
        if not line.strip():
            continue
        try:
            rows.append(json.loads(line))
        except ValueError as exc:
            raise ValueError(f'{path.name}:{i}: malformed JSON') from exc
    return rows


def verify(root):
    spec = json.loads((root/'experiments.json').read_text(encoding='utf8'))
    manifest = json.loads((root/'files.json').read_text(encoding='utf8'))
    files = {}
    question_orders = {}
    for entry in manifest['files']:
        if entry['path'] in files:
            raise ValueError(f'Duplicate manifest path: {entry["path"]}')
        rows = read_jsonl(safe_path(root,entry['path']),entry['sha256'])
        if len(rows) != entry['rows']:
            raise ValueError(f'Row count changed: {entry["path"]}')
        if entry['role'] in ('answers','queries'):
            questions = [r.get('query',r.get('input')) for r in rows]
            if len(rows) != DATASETS[entry['dataset']] or any(not isinstance(q,str) or not q.strip() for q in questions):
                raise ValueError(f'Invalid question records: {entry["path"]}')
            if len(set(questions)) != len(questions):
                raise ValueError(f'Duplicate questions: {entry["path"]}')
            reference = question_orders.setdefault(entry['dataset'],questions)
            if reference != questions:
                raise ValueError(f'Question order differs within dataset: {entry["path"]}')
            if entry['role'] == 'answers' and any(not isinstance(r.get('answer'),str) or not r['answer'].strip() for r in rows):
                raise ValueError(f'Empty answer: {entry["path"]}')
        files[entry['path']] = rows
    packaged = {p.relative_to(root).as_posix() for p in (root/'raw').rglob('*.jsonl')}
    if packaged != set(files):
        raise ValueError('Raw artifacts and file manifest do not match')
    comparisons, grouped = [], defaultdict(list)
    seen = set()
    for entry in spec['experiments']:
        key = (entry['id'],entry['dataset'])
        if key in seen:
            raise ValueError(f'Duplicate comparison: {key}')
        seen.add(key)
        values = summarize(files[entry['evaluation_file']],DATASETS[entry['dataset']],entry['target_first_pass_answer'])
        for metric,expected in entry['expected_percent'].items():
            if Decimal(values[metric]['percent']) != Decimal(str(expected)):
                raise ValueError(f'Paper mismatch: {key}, {metric}')
        comparisons.append({'id':entry['id'],'dataset':entry['dataset'],'metrics':values})
        grouped[entry['id']].append((entry['dataset'],values['Overall Winner']))
        # These are candidate archive inputs, not a claim that the judgment
        # records embed the answer hashes or establish execution provenance.
        for pair in entry.get('candidate_answer_pairs',[]):
            left,right = (files[p] for p in pair)
            if [r.get('query',r.get('input')) for r in left] != [r.get('query',r.get('input')) for r in right]:
                raise ValueError(f'Candidate answer question-order mismatch: {key}')
    aggregates = []
    for identity, items in grouped.items():
        if {ds for ds,_ in items} != set(DATASETS) or len(items) != 4:
            raise ValueError(f'Incomplete four-dataset experiment: {identity}')
        actual = aggregate([v for _,v in items])
        for k,v in spec['aggregate_expectations'].get(identity,{}).items():
            if Decimal(actual[k]) != Decimal(str(v)):
                raise ValueError(f'Aggregate paper mismatch: {identity}, {k}')
        aggregates.append({'id':identity,**actual})
    return {'files_verified': len(files), 'comparisons_verified': len(comparisons),
            'paper_percentages_verified': sum(len(e['expected_percent']) for e in spec['experiments'])
              +sum(len(e) for e in spec['aggregate_expectations'].values()),
            'comparisons':comparisons,'aggregates':aggregates,
            'boundary':'Numerical and file-integrity verification, not verification of historical model identity or bitwise regeneration.'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,help='Optional generated report JSON (never an input path)')
    args = parser.parse_args()
    root = Path(__file__).resolve().parent
    if args.output and (args.output.resolve().is_relative_to(root/'raw') or args.output.resolve() in {root/'experiments.json',root/'files.json'}):
        parser.error('Output must not overwrite an input artifact')
    result = verify(root)
    if args.output:
        args.output.write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf8')
    print(f"PASS: {result['files_verified']} files, {result['comparisons_verified']} comparisons, {result['paper_percentages_verified']} paper percentages")
    for item in result['aggregates']:
        print(f"{item['id']}: macro={item['macro']}%, pooled={item['pooled']}%")
    print(result['boundary'])


if __name__ == '__main__':
    main()
