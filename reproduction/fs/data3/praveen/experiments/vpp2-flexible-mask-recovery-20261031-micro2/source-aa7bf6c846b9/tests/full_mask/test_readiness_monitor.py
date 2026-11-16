import copy
import unittest
from vpp2.full_mask.monitor import confirm_full
from vpp2.full_mask.review_decisions import validate_review


class Readiness(unittest.TestCase):
    def test_quick_stability_cannot_bypass_full_suite_checks(self):
        c={'train':dict(hidden_churn_max=.01,ffn_churn_max=.03,
            suite_error_relative_change_max=.02,suite_sensitivity_top_overlap_min=.9)}
        record=dict(full=True,selection=dict(hidden_indices=[0,1],ffn_indices=[[0,1],[0,1]]),
                    by_suite={s:dict(hard_error=1.,hidden_top=[0,1],ffn_top=[[0,1]],
                                    half_hidden_overlap=1.,half_ffn_overlap=[1.])
                              for s in ('memory','precision')})
        self.assertFalse(confirm_full(None,record,c)['passed'])
        self.assertTrue(confirm_full(record,copy.deepcopy(record),c)['passed'])
        bad=copy.deepcopy(record);bad['by_suite']['precision']['half_hidden_overlap']=.5
        self.assertFalse(confirm_full(record,bad,c)['passed'])
        bad=copy.deepcopy(record);bad['by_suite']['memory']['hard_error']=1.1
        self.assertFalse(confirm_full(record,bad,c)['passed'])

    def test_review_requires_evidence_and_does_not_delete_uncertainty(self):
        row=dict(window_id='x',episode=1,frame=0,role='selection_initial',protected_suite=True)
        panel=dict(identity='test',windows=[row])
        review=dict(panel_identity='test',decisions=[])
        self.assertFalse(validate_review(panel,review)['passed'])
        decision=dict(window_id='x',episode=1,frame=0,decision='retain_verified',evidence='')
        review['decisions']=[decision]
        self.assertFalse(validate_review(panel,review)['passed'])
        decision['evidence']='Specific cue visible in the native history at frame 0.'
        self.assertTrue(validate_review(panel,review)['passed'])
        decision['decision']='retain_uncertain'
        r=validate_review(panel,review)
        self.assertFalse(r['passed']);self.assertEqual(r['automatic_deletions'],0)
        decision['decision']='suggest_duplicate'
        self.assertEqual(validate_review(panel,review)['requested_replacements'],['x'])


if __name__=='__main__':unittest.main()
