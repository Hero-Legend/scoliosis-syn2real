"""Portable runtime adapter for the unchanged training/evaluation functions."""
from __future__ import annotations
import argparse, hashlib, json
from pathlib import Path
from prepare_data import digest, rows, validate_metadata

ROOT=Path(__file__).resolve().parent
METHODS=('C1_R1_REAL_ONLY_CE','C1_R2_SYN_PRETRAIN_CE','C1_R3_NAIVE_JOINT_CE')

def build_plan(config, budgets):
    import math
    training=config['training']
    def steps(n):
        return training['epochs']*math.ceil(math.ceil(n/training['micro_target_batch_size'])/training['gradient_accumulation_steps'])
    source=dict(job_id='C1-SOURCE-PRETRAIN-CE',job_type='SOURCE_PRETRAIN',method_id=METHODS[1],
                budget_percent='NA',label_budget_seed='NA',training_seed=2026082101,
                checkpoint_dependency='NONE',source_examples_per_epoch=15999,
                target_examples_per_epoch=0,maximum_optimizer_steps=steps(15999))
    plan=[source]
    for budget in config['budgets_percent']:
        for seed in config['label_budget_seeds']:
            n=sum(r['label_visible']=='1' for r in budgets if int(r['label_budget_seed'])==seed and int(r['budget_percent'])==budget)
            if n==0:
                raise ValueError('Empty budget cell')
            for method in METHODS:
                plan.append(dict(job_id=f'C1-{method}-B{budget}-S{seed}',job_type='TARGET_FIT',method_id=method,
                    budget_percent=str(budget),label_budget_seed=str(seed),training_seed=seed,
                    checkpoint_dependency='C1_SYN_CE_PRETRAIN_V1' if method==METHODS[1] else 'DINOv2_OFFICIAL_INIT',
                    source_examples_per_epoch=n if method==METHODS[2] else 0,
                    target_examples_per_epoch=n,maximum_optimizer_steps=steps(n)))
    return plan

def bind(workspace, weight):
    from engine import training
    config=json.loads((ROOT/'configs/training.json').read_text())
    if digest(weight)!=config['backbone']['official_weight_sha256']:
        raise ValueError('Wrong DINOv2 initialization file')
    config['backbone']['official_weight_path']=str(weight.resolve())
    real=rows(ROOT/'data/aasce_identity.csv'); source=rows(ROOT/'data/spinalai_identity.csv')
    budgets=rows(ROOT/'data/label_budgets.csv')
    validate_metadata(real,source,budgets)
    index_path=workspace/'inputs/training.json'
    index=json.loads(index_path.read_text())
    expected={split:{r['case_id']:r for r in real if r['dataset_split']==split} for split in ('TRAINDEV','VALIDATION')}
    expected['SOURCE']={r['case_id']:r for r in source}
    if set(index)!=set(expected):
        raise ValueError('Unexpected training input partitions')
    for split, reference in expected.items():
        loaded=index[split]
        if len(loaded)!=len(reference) or {r['case_id'] for r in loaded}!=set(reference):
            raise ValueError('Training identity mismatch')
        for r in loaded:
            if r['severity_grade'] not in (0,1,2) or r['image_sha256']!=reference[r['case_id']]['image_sha256']:
                raise ValueError('Invalid label or stored image identity')
            if digest(Path(r['image_path']))!=r['image_sha256']:
                raise ValueError('Dataset image changed after preparation')
    plan=build_plan(config,budgets)
    def select(config,job):
        if job['job_type']=='SOURCE_PRETRAIN':
            return [],[],index['SOURCE']
        visible={r['case_id'] for r in budgets if r['label_budget_seed']==job['label_budget_seed'] and r['budget_percent']==job['budget_percent'] and r['label_visible']=='1'}
        return [r for r in index['TRAINDEV'] if r['case_id'] in visible],index['VALIDATION'],index['SOURCE'] if job['method_id']==METHODS[2] else []
    training.ROOT=workspace
    training.PREFLIGHT_PATH=workspace/'preflight.json'
    training.select_job_data=select
    # New local execution fingerprint, not a fabricated historical result lock.
    files=[ROOT/'run.py',ROOT/'prepare_data.py',ROOT/'configs/training.json',index_path]
    files+=sorted((ROOT/'engine').glob('*.py'))+sorted((ROOT/'data').glob('*.csv'))
    contract=hashlib.sha256(''.join(digest(p) for p in files).encode()).hexdigest()
    return training,config,plan,contract

def evaluate_heldout(training,config,plan,contract,workspace,job_id):
    import torch
    from engine import model
    jobs=[j for j in plan if j['job_id']==job_id and j['job_type']=='TARGET_FIT']
    if len(jobs)!=1:
        raise ValueError('Evaluation needs a target job ID')
    # Prevent evaluation while the planned model matrix is still incomplete.
    for j in plan:
        completion=workspace/'runs/c1'/j['job_id']/'completion.json'
        if not completion.is_file() or json.loads(completion.read_text())['contract_digest']!=contract:
            raise ValueError('Finish all planned training jobs before held-out evaluation')
    output=workspace/'evaluation'/job_id
    if output.exists():
        raise FileExistsError('Evaluation output already exists; refusing to overwrite')
    identities={r['case_id']:r for r in rows(ROOT/'data/aasce_identity.csv') if r['dataset_split']=='LOCKED_BENCHMARK'}
    data=json.loads((workspace/'inputs/heldout.json').read_text())
    if len(data)!=120 or {r['case_id'] for r in data}!=set(identities):
        raise ValueError('Held-out identity mismatch')
    for r in data:
        if r['severity_grade'] not in (0,1,2) or digest(Path(r['image_path']))!=identities[r['case_id']]['image_sha256']:
            raise ValueError('Held-out label or image mismatch')
    job=jobs[0]; seed=int(job['training_seed'])
    net=model.build_torch_components(config,'softmax_cross_entropy',seed).to('cuda')
    checkpoint=torch.load(workspace/'runs/c1'/job_id/'final.pt',map_location='cpu',weights_only=False)
    if checkpoint['contract_digest']!=contract:
        raise ValueError('Checkpoint belongs to a different execution')
    net.load_state_dict(checkpoint['model_state'],strict=True)
    loader=model.make_eval_loader(model.make_dataset(data,config,False,seed),config['training']['evaluation_batch_size'],config['training']['num_workers'])
    metrics,predictions=training.evaluate(net,loader,torch.device('cuda'),config['evaluation']['ece_equal_width_bins'],config)
    output.mkdir(parents=True,exist_ok=False)
    training.atomic_json(output/'metrics.json',metrics)
    training._write_predictions(output/'predictions.csv',predictions)

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action',choices=['plan','preflight','train','evaluate'])
    parser.add_argument('--workspace',type=Path)
    parser.add_argument('--weights',type=Path)
    parser.add_argument('--job-id')
    args=parser.parse_args()
    if args.action=='plan':
        config=json.loads((ROOT/'configs/training.json').read_text())
        plan=build_plan(config,rows(ROOT/'data/label_budgets.csv'))
        for job in plan:
            print(job['job_id'])
        return
    if not args.workspace or not args.weights:
        parser.error('Provide --workspace and --weights')
    training,config,plan,contract=bind(args.workspace.resolve(),args.weights)
    if args.action=='preflight':
        if training.run_preflight(config,plan,contract)['decision']!='PASS':
            raise SystemExit(2)
    elif args.action=='train':
        jobs=[j for j in plan if j['job_id']==args.job_id]
        if len(jobs)!=1:
            parser.error('Provide a job ID from the plan')
        training.run_job(config,jobs[0],contract)
    else:
        evaluate_heldout(training,config,plan,contract,args.workspace.resolve(),args.job_id)

if __name__=='__main__':
    main()
