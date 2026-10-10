import unittest
import pandas as pd
from retail.diagnose import customer_bridge, price_mix_bridge, product_shapley


class DiagnosisTests(unittest.TestCase):
    def test_aov_shapley_hand_calculation(self):
        # AOV rises 2*3 -> 4*5: units contributes 2*4=8, price 2*3=6.
        got=product_shapley(2,3,4,5)
        self.assertEqual(got['delta'],14)
        self.assertEqual(got['quantity_component'],8)
        self.assertEqual(got['value_component'],6)
        self.assertEqual(got['reconciliation_error'],0)

    def test_price_mix_with_entry_exit_hand_calculation(self):
        # Common A: p 2->4; common units share 1/2->1/4. Exit B p6,
        # enter C p8. Total unit value 4->7, bridge .75- .75+6-3=3.
        d=pd.DataFrame([['A',10,20,5,20],['B',10,60,0,0],['C',0,0,15,120]],
            columns=['key','units_0','revenue_0','units_1','revenue_1'])
        got,_=price_mix_bridge(d)
        expected={'within_common_key_paid_unit_value':.75,'common_key_quantity_mix':0,
            'common_set_quantity_share':-.75,'entering_key_revenue_per_total_unit':6,
            'exiting_key_revenue_per_total_unit':-3}
        self.assertEqual(got['components'],expected)
        self.assertEqual(got['delta'],3)
        self.assertEqual(got['reconciliation_error'],0)

    def test_same_price_mix_is_not_price_effect(self):
        d=pd.DataFrame([['A',10,20,30,60],['B',30,240,10,80]],
            columns=['key','units_0','revenue_0','units_1','revenue_1'])
        got,detail=price_mix_bridge(d)
        self.assertEqual(got['delta'],-3)
        self.assertEqual(got['components']['within_common_key_paid_unit_value'],0)
        self.assertEqual(got['components']['common_key_quantity_mix'],-3)
        self.assertEqual(detail.common_key_mix_contribution_centered.tolist(),[-1.5,-1.5])

    def test_disjoint_and_empty_sets(self):
        cols=['key','units_0','revenue_0','units_1','revenue_1']
        got,_=price_mix_bridge(pd.DataFrame([['A',2,6,0,0],['B',0,0,4,20]],columns=cols))
        self.assertEqual(got['common_keys'],0)
        self.assertEqual(got['delta'],2)
        self.assertEqual(got['reconciliation_error'],0)
        got,_=price_mix_bridge(pd.DataFrame(columns=cols))
        self.assertEqual(got['status'],'undefined_no_positive_units_in_one_period')

    def test_customer_entry_is_not_all_acquisition(self):
        d=pd.DataFrame([
            ['A','2010-01-01',1,1,10,15],['B','2010-11-01',1,0,20,0],
            ['C','2011-11-03',0,1,0,8],['D','2011-05-01',0,1,0,9]],
            columns=['customer_id','first_day','orders_0','orders_1','revenue_0','revenue_1'])
        got=customer_bridge(d).set_index('customer_id')
        self.assertEqual(got.loc['C','customer_group'],'current_only_first_observed_in_november')
        self.assertEqual(got.loc['D','customer_group'],'current_only_previously_observed')
        self.assertEqual(got.revenue_delta.sum(),2)
        self.assertEqual(got.groupby('customer_group').revenue_delta.sum().sum(),2)


if __name__=='__main__':
    unittest.main()
