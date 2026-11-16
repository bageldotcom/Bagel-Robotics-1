import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
from vpp2.full_mask.panel import digest, file_hash
from vpp2.critical_windows.staged_review import select_initial, validate_record, assess, import_record, load_records, SCHEMA
from vpp2.critical_windows.selection import choose, CoverageConflict


class StagedReview(unittest.TestCase):
    def test_initial_choices_are_task_balanced_fitting_only_and_metadata_only(self):
        episodes = [dict(episode=i, task='a', role='fit', length=i*10) for i in range(1,5)]
        episodes += [dict(episode=i, task='b', role='fit', length=i*10) for i in range(5,9)]
        episodes += [dict(episode=0, task='a', role='monitor', length=1)]
        manifest = dict(tasks=['b','a'], episodes=episodes)
        self.assertEqual(select_initial(manifest), [6,2])
        for e in episodes: e['policy_error'] = -e['episode']
        self.assertEqual(select_initial(manifest), [6,2])

    def fixture(self, root):
        ep = dict(episode=1, task='task', role='fit', suite='memory', length=64)
        packet_root = root/'packets/00001'; packet_root.mkdir(parents=True)
        page = packet_root/'coarse_000.jpg'; page.write_bytes(b'fixture image bytes')
        packet = dict(episode=1, coarse_pages=[dict(file=page.name,sha256=file_hash(page),frames=[0,24,48,63])])
        packet['identity']=digest(packet); (packet_root/'packet.json').write_text(json.dumps(packet))
        stage=dict(identity='stage')
        record=dict(schema=SCHEMA,stage_identity='stage',episode=1,task='task',packet_identity=packet['identity'],
            status='screened',reviewer='test reviewer',full_coarse_timeline_inspected=True,
            inspected_pages={page.name:file_hash(page)},observations='Visible hand movement.',
            critical_interval_boundaries_verified=False,manipulation_anchor_frames=[24],
            memory_cue_status='uncertain',findings=[])
        return dict(root=str(root)),dict(episodes=[ep]),stage,record

    def test_screen_cannot_fake_unseen_frames_labels_or_evidence(self):
        with tempfile.TemporaryDirectory() as td:
            config,manifest,stage,record=self.fixture(Path(td))
            validate_record(config,manifest,stage,record)
            for key,value in [('inspected_pages',{}),('manipulation_anchor_frames',[23]),
                              ('critical_interval_boundaries_verified',True),('memory_cue_status','identified')]:
                bad=copy.deepcopy(record);bad[key]=value
                with self.assertRaises(ValueError):validate_record(config,manifest,stage,bad)
            (Path(td)/'packets/00001/coarse_000.jpg').write_bytes(b'changed')
            with self.assertRaises(ValueError):validate_record(config,manifest,stage,record)

    def test_monitor_screen_cannot_change_selection(self):
        with tempfile.TemporaryDirectory() as td:
            config,manifest,stage,record=self.fixture(Path(td));manifest['episodes'][0]['role']='monitor'
            with self.assertRaises(ValueError):validate_record(config,manifest,stage,record)

    def test_missing_phase_expands_review_and_uncovered_anchor_fails_final_gate(self):
        manifest=dict(episodes=[dict(episode=i,task='a',role='fit',length=100+i) for i in range(1,5)])
        stage=dict(identity='stage',base_panel_identity='base',required_episodes=[2],tasks=['a'])
        record=dict(task='a',manipulation_anchor_frames=[24],findings=[dict(id='gap',kind='missing_phase',frames=[24],evidence='Visible grasp absent.')])
        with patch('vpp2.critical_windows.staged_review.prepare',return_value=stage), \
             patch('vpp2.critical_windows.staged_review.read_json',return_value=manifest), \
             patch('vpp2.critical_windows.staged_review.checked_identity',side_effect=lambda x:x), \
             patch('vpp2.critical_windows.staged_review.load_records',return_value={2:record}):
            panel=dict(identity='base',windows=[dict(role='selection_initial',episode=2,action_interval=[32,64])])
            result,_=assess(dict(root='/unused'),panel)
            self.assertEqual(result['missing_episodes'],[1]);self.assertFalse(result['passed'])
            self.assertEqual(result['unresolved_manipulation_anchors'],[[2,24]])

    def test_coarse_anchor_is_preserved_without_inventing_dense_interval(self):
        rows=[dict(episode=1,frame=t,length=96,role='selection_initial',temporal_quarter=t//24,
                   relative_time=t/64,proxy_tags=[]) for t in range(0,65,8)]
        selected,_=choose(rows,np.arange(len(rows))[:,None],2,7,3,{},base=0,random_fraction=0,
                          observed_anchors={1:dict(role='fit',frames=[20,60])})
        for t in (20,60):self.assertTrue(any(rows[i]['frame']<=t<rows[i]['frame']+32 for i in selected))
        self.assertGreaterEqual(rows[selected[1]]['frame']-rows[selected[0]]['frame'],32)
        with self.assertRaises(ValueError):
            choose(rows,np.arange(len(rows))[:,None],2,7,3,{},base=0,random_fraction=0,
                   observed_anchors={1:dict(role='monitor',frames=[20])})

    def test_anchor_budget_conflict_is_explicit(self):
        rows=[dict(episode=1,frame=t,length=128,role='selection_initial',temporal_quarter=t//32,
                   relative_time=t/96,proxy_tags=[]) for t in range(0,97,8)]
        with self.assertRaises(CoverageConflict):
            choose(rows,np.arange(len(rows))[:,None],1,7,1,{},base=0,random_fraction=0,
                   observed_anchors={1:dict(role='fit',frames=[0,127])})


    def test_revised_screen_requires_previous_identity_and_keeps_history(self):
        with tempfile.TemporaryDirectory() as td:
            root=Path(td);config,manifest,stage,record=self.fixture(root)
            manifest['identity']=digest(manifest)
            (root/'audit_manifest.json').write_text(json.dumps(manifest))
            candidate=root/'input.json';candidate.write_text(json.dumps(record))
            with patch('vpp2.critical_windows.staged_review.prepare',return_value=stage):
                import_record(config,candidate)
                revised={**record,'observations':'Re-inspected; a later visible grasp is also relevant.',
                         'manipulation_anchor_frames':[24,48]}
                candidate.write_text(json.dumps(revised))
                with self.assertRaises(ValueError):import_record(config,candidate)
                self.assertEqual(load_records(config,manifest,stage)[1],record)
                revised['supersedes_record_identity']=digest(record)
                candidate.write_text(json.dumps(revised));import_record(config,candidate)
                self.assertEqual(load_records(config,manifest,stage)[1],revised)
                self.assertEqual(len(list((root/'staged_review/history').glob('*.json'))),2)
                import_record(config,candidate)  # Idempotent replay cannot add revisions.
                self.assertEqual(len(list((root/'staged_review/history').glob('*.json'))),2)

    def test_publication_rechecks_dense_labels_anchors_and_monitor_choices(self):
        from vpp2.critical_windows.reselect import validate_publication
        def row(ep,t,role):
            return dict(episode=ep,frame=t,role=role,task='task',length=192,
                        action_interval=[t,t+32],content_sha256=f'{ep}:{t}',history_indices=[0,t],
                        selection_reason='balanced_random_coverage' if t==0 else 'additional_distinct_behavior')
        rows=[row(1,t,'selection_initial') for t in (0,32,64,96)]+[row(2,0,'monitor')]
        original=dict(windows=copy.deepcopy(rows))
        config=dict(critical_selection=dict(target_total=4,random_fraction=.25),
                    curation=dict(max_windows_per_episode=4))
        annotations={1:dict(role='fit',intervals=[dict(frames=[100,106])])}
        anchors={1:dict(role='fit',frames=[95])}
        report=validate_publication(config,original,rows,annotations,anchors)
        self.assertEqual(report['verified_critical_action_steps'],6)
        self.assertEqual(report['observed_task_relevant_frames'],1)
        lost_dense=copy.deepcopy(rows);lost_dense[3]=row(1,128,'selection_initial')
        with self.assertRaisesRegex(ValueError,'critical action steps'):
            validate_publication(config,original,lost_dense,annotations,anchors)
        lost_anchor=copy.deepcopy(rows);lost_anchor[2]=row(1,128,'selection_initial')
        with self.assertRaisesRegex(ValueError,'task-relevant frame'):
            validate_publication(config,original,lost_anchor,annotations,anchors)
        changed_monitor=copy.deepcopy(rows);changed_monitor[-1]=row(2,32,'monitor')
        with self.assertRaisesRegex(ValueError,'Monitoring window choices'):
            validate_publication(config,original,changed_monitor,annotations,anchors)
        overlap=copy.deepcopy(rows);overlap[1]=row(1,16,'selection_initial')
        with self.assertRaisesRegex(ValueError,'overlapping'):
            validate_publication(config,original,overlap,annotations,anchors)



if __name__=='__main__':unittest.main()
