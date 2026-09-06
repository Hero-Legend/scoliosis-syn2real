"""Reconstruct local input labels from upstream files; no data are uploaded."""
from __future__ import annotations
import argparse, csv, hashlib, json, math
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent

def rows(path):
    with Path(path).open(encoding='utf-8-sig', newline='') as handle:
        return list(csv.DictReader(handle))

def raw_rows(path):
    with Path(path).open(encoding='utf-8-sig', newline='') as handle:
        return list(csv.reader(handle))

def digest(path):
    value = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(1024*1024), b''):
            value.update(block)
    return value.hexdigest()

def grade(angles):
    values = [float(x) for x in angles]
    if len(values) != 3 or any(not math.isfinite(x) or x < 0 for x in values):
        raise ValueError('Expected three finite non-negative angle labels')
    return int(max(values) >= 30) + int(max(values) >= 45)

def safe_child(root, relative):
    root = Path(root).resolve()
    path = (root/relative).resolve()
    if not path.is_relative_to(root):
        raise ValueError('Input path leaves its dataset directory')
    return path

def validate_metadata(real, source, budgets):
    if len(real) != 480 or len({r['case_id'] for r in real}) != 480:
        raise ValueError('Expected 480 unique AASCE identities')
    if len({r['source_row'] for r in real}) != 480:
        raise ValueError('Duplicate original source row')
    if len(source) != 15999 or len({r['case_id'] for r in source}) != 15999:
        raise ValueError('Expected 15999 unique synthetic identities')
    counts = Counter(r['dataset_split'] for r in real)
    if counts != {'TRAINDEV':288,'VALIDATION':72,'LOCKED_BENCHMARK':120}:
        raise ValueError('Unexpected split sizes')
    groups = defaultdict(set)
    image_splits = defaultdict(set)
    for r in real:
        groups[r['containment_group']].add(r['dataset_split'])
        image_splits[r['image_sha256']].add(r['dataset_split'])
    if len(groups) != 229 or any(len(v) != 1 for v in [*groups.values(),*image_splits.values()]):
        raise ValueError('Group or exact-image overlap between splits')
    if {r['image_sha256'] for r in real} & {r['image_sha256'] for r in source}:
        raise ValueError('Exact source-target image overlap')
    train = {r['case_id']:r for r in real if r['dataset_split']=='TRAINDEV'}
    cells=defaultdict(list)
    for r in budgets:
        if r['case_id'] not in train or r['label_visible'] not in ('0','1'):
            raise ValueError('Invalid budget identity or visibility')
        cells[(int(r['label_budget_seed']),int(r['budget_percent']))].append(r)
    seeds=list(range(2026081901,2026081906))
    if set(cells) != {(s,b) for s in seeds for b in (10,25,50,100)}:
        raise ValueError('Unexpected seed-budget cells')
    for seed in seeds:
        previous=set()
        for budget in (10,25,50,100):
            cell=cells[seed,budget]
            if len(cell)!=288 or {r['case_id'] for r in cell}!=set(train):
                raise ValueError('Incomplete/duplicate budget cell')
            visible={r['case_id'] for r in cell if r['label_visible']=='1'}
            if not previous.issubset(visible) or (budget==100 and visible!=set(train)):
                raise ValueError('Non-nested budgets')
            group_flags=defaultdict(set)
            for r in cell:
                group_flags[train[r['case_id']]['containment_group']].add(r['label_visible'])
            if any(len(v)!=1 for v in group_flags.values()):
                raise ValueError('A containment group was partially selected')
            previous=visible

def attach(identity, path, angles):
    if digest(path) != identity['image_sha256']:
        raise ValueError('Image identity mismatch: '+identity['case_id'])
    return dict(case_id=identity['case_id'],image_path=str(path),image_sha256=identity['image_sha256'],
                containment_group=identity.get('containment_group',identity['case_id']),severity_grade=grade(angles))

def prepare(aasce, spinalai, workspace):
    real=rows(ROOT/'data/aasce_identity.csv')
    source=rows(ROOT/'data/spinalai_identity.csv')
    budgets=rows(ROOT/'data/label_budgets.csv')
    validate_metadata(real,source,budgets)
    names=raw_rows(aasce/'labels/training/filenames.csv')
    angles=raw_rows(aasce/'labels/training/angles.csv')
    if len(names)!=481 or len(angles)!=481 or any(len(r)!=1 for r in names) or len({r[0] for r in names})!=481:
        raise ValueError('Expected the original aligned 481-row AASCE training labels')
    synthetic_labels=raw_rows(spinalai/'Cobb_spinal-AI2024-train_gt.txt')
    if any(len(r)!=4 for r in synthetic_labels):
        raise ValueError('Expected filename and three comma-separated synthetic angles')
    by_name={r[0]:r[1:] for r in synthetic_labels}
    if len(by_name)!=len(synthetic_labels):
        raise ValueError('Duplicate synthetic label filenames')
    train={'TRAINDEV':[],'VALIDATION':[],'SOURCE':[]}
    heldout=[]
    for r in real:
        i=int(r['source_row'])
        path=safe_child(aasce/'data/training',names[i][0])
        record=attach(r,path,angles[i])
        if r['dataset_split']=='LOCKED_BENCHMARK':
            heldout.append(record)
        else:
            train[r['dataset_split']].append(record)
    for r in source:
        path=safe_child(spinalai,Path(r['subset'])/r['filename'])
        train['SOURCE'].append(attach(r,path,by_name[r['filename']]))
    destination=workspace.resolve()/'inputs'
    destination.mkdir(parents=True,exist_ok=False)
    # Keep held-out labels in a separate file; training never reads this file.
    for name,payload in [('training.json',train),('heldout.json',heldout)]:
        (destination/name).write_text(json.dumps(payload,indent=2)+'\n',encoding='utf-8')
    print('Prepared local inputs; no training or network upload was performed.')

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--check-metadata',action='store_true')
    parser.add_argument('--aasce-root',type=Path)
    parser.add_argument('--spinalai-root',type=Path)
    parser.add_argument('--workspace',type=Path)
    args=parser.parse_args()
    if args.check_metadata:
        validate_metadata(rows(ROOT/'data/aasce_identity.csv'),rows(ROOT/'data/spinalai_identity.csv'),rows(ROOT/'data/label_budgets.csv'))
        print('PASS: identities, split isolation, paired cells and group-complete nested budgets')
        return
    if not all([args.aasce_root,args.spinalai_root,args.workspace]):
        parser.error('Provide --aasce-root, --spinalai-root and --workspace')
    prepare(args.aasce_root,args.spinalai_root,args.workspace)

if __name__=='__main__':
    main()
