import unittest
import numpy as np
from campaign.stats import dr_scores,policy_scores,targeting,fixed_random,holm,estimate,sample_size
from campaign.models import FEATURES


class CampaignTests(unittest.TestCase):
    def test_dr_exact_nuisance_balanced_rct(self):
        actions=np.tile([0,1,2],100)
        mu=np.tile([1.,3.,2.],(300,1));y=mu[np.arange(300),actions]
        scores=dr_scores(y,actions,mu)
        self.assertAlmostEqual(policy_scores(fixed_random(300,1,1),scores).mean(),2.)
        self.assertAlmostEqual(policy_scores(fixed_random(300,1,.25),scores).mean(),.5)

    def test_no_contact_exact_zero_even_with_noise(self):
        rng=np.random.default_rng(1); scores=rng.normal(size=(100,3))
        self.assertTrue(np.all(policy_scores(fixed_random(100,1,0),scores)==0))

    def test_budget_and_negative_scores(self):
        p=targeting(np.tile([1.,2.],(100,1)),.15625)
        self.assertEqual(np.sum(p[:,0]==0),15)
        self.assertTrue(np.all(p[p[:,0]==0,2]==1))
        p=targeting(np.full((100,2),-1.),1)
        self.assertTrue(np.all(p[:,0]==1))

    def test_known_probability_ht(self):
        psi=dr_scores([0,3,0],[0,1,2],np.zeros((3,3)))
        self.assertAlmostEqual(policy_scores(fixed_random(3,1,1),psi).mean(),3)

    def test_holm_order_and_monotonicity(self):
        np.testing.assert_allclose(holm([.04,.001,.02]),[.04,.003,.04])

    def test_feature_contract(self):
        self.assertTrue(set(FEATURES).isdisjoint({'treatment','visit','conversion','spend','customer_id'}))

    def test_cost_pairing_same_users(self):
        x=np.arange(100.)
        self.assertEqual(estimate(x-x)['estimate'],0)
        self.assertGreater(sample_size(.006,.002),sample_size(.006,.004))


if __name__=='__main__':unittest.main()
