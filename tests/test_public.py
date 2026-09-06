import copy, json, tempfile, unittest
import builtins, dis, inspect, types
from unittest.mock import patch
from pathlib import Path
import numpy as np
import prepare_data as data
from run import ROOT, build_plan
from engine import utils, training
from engine import model, statistics, sensitivity

class PublicReleaseTests(unittest.TestCase):
    def metadata(self):
        return [data.rows(ROOT/'data'/name) for name in ['aasce_identity.csv','spinalai_identity.csv','label_budgets.csv']]

    def test_metadata(self):
        data.validate_metadata(*self.metadata())

    def test_label_boundaries(self):
        self.assertEqual([data.grade([0,0,x]) for x in [29.999,30,44.999,45]],[0,1,1,2])

    def test_nonfinite_rejected(self):
        for values in [[0,1,float('nan')],[0,1,-1],[0,1]]:
            with self.assertRaises(ValueError): data.grade(values)

    def test_path_escape_rejected(self):
        with tempfile.TemporaryDirectory() as path:
            with self.assertRaises(ValueError): data.safe_child(path,'../outside.jpg')

    def test_duplicate_identity_rejected(self):
        real,source,budgets=self.metadata()
        real[0]=copy.deepcopy(real[1])
        with self.assertRaises(ValueError): data.validate_metadata(real,source,budgets)

    def test_partial_group_rejected(self):
        real,source,budgets=self.metadata()
        counts={}
        for row in real:
            if row['dataset_split']=='TRAINDEV': counts.setdefault(row['containment_group'],[]).append(row['case_id'])
        member=next(v[0] for v in counts.values() if len(v)>1)
        row=next(r for r in budgets if r['case_id']==member)
        row['label_visible']=str(1-int(row['label_visible']))
        with self.assertRaises(ValueError): data.validate_metadata(real,source,budgets)

    def test_plan_pairing(self):
        config=json.loads((ROOT/'configs/training.json').read_text())
        plan=build_plan(config,self.metadata()[2])
        self.assertEqual(len(plan),61)
        for budget in config['budgets_percent']:
            for seed in config['label_budget_seeds']:
                cell=[j for j in plan if j['budget_percent']==str(budget) and j['label_budget_seed']==str(seed)]
                self.assertEqual(len(cell),3)
                self.assertEqual(len({j['maximum_optimizer_steps'] for j in cell}),1)
                self.assertEqual(len({j['target_examples_per_epoch'] for j in cell}),1)

    def test_metrics_on_artificial_fixture(self):
        truth=np.array([0,1,2])
        self.assertEqual(utils.classification_metrics(truth,truth)['qwk'],1)
        self.assertEqual(training.reliability_metrics(truth,np.eye(3),10),dict(ece=0.,brier=0.))

    def test_amp_guard(self):
        self.assertTrue(utils.amp_optimizer_step_succeeded(8,8))
        self.assertFalse(utils.amp_optimizer_step_succeeded(8,4))

    def test_optimizer_budget(self):
        self.assertEqual(training.expected_optimizer_steps(33,80,4,8),160)

    def test_extracted_global_dependencies(self):
        def inspect_code(code, namespace):
            for instruction in dis.get_instructions(code):
                if instruction.opname=='LOAD_GLOBAL':
                    self.assertTrue(instruction.argval in namespace or hasattr(builtins,instruction.argval),str(instruction.argval))
            for value in code.co_consts:
                if isinstance(value,types.CodeType): inspect_code(value,namespace)
        for module in (utils,model,training,statistics,sensitivity):
            for name,value in vars(module).items():
                if inspect.isfunction(value) and value.__module__==module.__name__:
                    inspect_code(value.__code__,vars(module))

    def test_adapter_uses_no_heldout_label_file(self):
        import run
        real,source,budgets=self.metadata()
        index={key:[] for key in ['TRAINDEV','VALIDATION','SOURCE']}
        hashes={}
        for original in [*real,*source]:
            split=original.get('dataset_split','SOURCE')
            if split=='LOCKED_BENCHMARK': continue
            virtual='virtual_'+original['case_id'].replace(':','_')+'.jpg'
            hashes[virtual]=original['image_sha256']
            index[split].append(dict(case_id=original['case_id'],image_path=virtual,
                image_sha256=original['image_sha256'],containment_group=original.get('containment_group',original['case_id']),severity_grade=0))
        expected=json.loads((ROOT/'configs/training.json').read_text())['backbone']['official_weight_sha256']
        genuine=run.digest
        def mocked_digest(path):
            if Path(path).name=='virtual_weights.pth': return expected
            if Path(path).name in hashes: return hashes[Path(path).name]
            return genuine(path)
        with tempfile.TemporaryDirectory() as folder:
            workspace=Path(folder); (workspace/'inputs').mkdir()
            (workspace/'inputs/training.json').write_text(json.dumps(index))
            with patch('run.digest',side_effect=mocked_digest):
                engine,config,plan,contract=run.bind(workspace,Path('virtual_weights.pth'))
            self.assertEqual(len(contract),64)
            target,validation,synthetic=engine.select_job_data(config,plan[0])
            self.assertEqual((len(target),len(validation),len(synthetic)),(0,0,15999))
            target,validation,synthetic=engine.select_job_data(config,plan[1])
            self.assertEqual(len(validation),72)
            self.assertFalse({r['case_id'] for r in target} & {r['case_id'] for r in real if r['dataset_split']=='LOCKED_BENCHMARK'})

    def test_cluster_bootstrap_keeps_whole_groups(self):
        indices=statistics.group_resample_indices(np.array(['a','a','b']),np.random.default_rng(12))
        self.assertEqual(np.sum(indices==0),np.sum(indices==1))

    def test_holm(self):
        self.assertTrue(np.allclose(statistics.holm_adjust([.01,.04,.03]),[.03,.06,.06]))

if __name__=='__main__': unittest.main()
