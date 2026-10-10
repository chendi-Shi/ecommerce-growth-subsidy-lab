import unittest
import numpy as np
from growth.data import generate, FEATURES
from growth.analysis import difference_ci, policy_rows, top_mask, required_sample_per_arm, uplift_curve


class AnalysisTests(unittest.TestCase):
    def test_ipw_full_population_denominator(self):
        import pandas as pd
        df=pd.DataFrame(dict(treatment=[1,0,1,0],propensity=[.5]*4,
            converted=[1,0,0,1],margin=[20,0,0,20],subsidy=[8,0,0,0]))
        conversion,margin,cost=policy_rows(df,np.array([1,1,0,0],dtype=bool))
        self.assertEqual(conversion.mean(),.5)
        self.assertEqual(margin.mean(),10)
        self.assertEqual(cost.mean(),4)

    def test_zero_variance(self):
        self.assertEqual(difference_ci([0,0],[0,0])['p_value'],1)

    def test_reproducible_and_no_leakage(self):
        a,_=generate(2000,42); b,_=generate(2000,42)
        self.assertTrue(a.equals(b))
        self.assertTrue(set(FEATURES).isdisjoint({'converted','treatment','margin','subsidy','true_uplift'}))
        self.assertTrue((a.loc[a.treatment==0,'subsidy']==0).all())
        self.assertTrue(a.user_id.is_unique)

    def test_targeting_budget_count(self):
        self.assertEqual(top_mask(np.arange(10),.3).sum(),3)
        self.assertEqual(top_mask(np.arange(10),0).sum(),0)
        self.assertGreater(required_sample_per_arm(mde=.01),required_sample_per_arm(mde=.02))

    def test_zero_effect_balanced_trial(self):
        import pandas as pd
        df=pd.DataFrame(dict(treatment=[1,0]*100,propensity=[.5]*200,
            converted=[1]*200,margin=[20]*200,subsidy=[0]*200))
        conversion,margin,cost=policy_rows(df,np.ones(200,dtype=bool))
        self.assertEqual(conversion.mean(),0)
        self.assertEqual(margin.mean(),0)

    def test_curve_endpoint_matches_itt_ht(self):
        frame,_=generate(2000,7)
        frame['uplift']=np.arange(len(frame))
        x,y,area=uplift_curve(frame)
        expected=policy_rows(frame,np.ones(len(frame),dtype=bool))[0].mean()
        self.assertAlmostEqual(y[-1],expected)
        self.assertEqual(x[0],0)
        self.assertEqual(x[-1],1)

    def test_public_log_chunked_unique_and_timezone(self):
        import tempfile
        from pathlib import Path
        from growth.data import behavior_metrics
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'events.csv'
            path.write_text('1,10,2,pv,1511568000\n1,10,2,pv,1511568000\n2,10,2,buy,1511568000\n')
            result=behavior_metrics(path,chunksize=1)
            pv=result[result.behavior=='pv'].iloc[0]
            self.assertEqual(pv.events,2)
            self.assertEqual(pv.users,1)
            self.assertEqual(pv.date,'2017-11-25')


if __name__=='__main__': unittest.main()
