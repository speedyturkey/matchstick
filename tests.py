import unittest
import pandas as pd
from matchstick import df_crossjoin, Matcher


class testCrossJoin(unittest.TestCase):

    def test_join_two_dataframes(self):
        df1, df2 = get_sample_data()
        crossed = df_crossjoin(df1, df2)
        self.assertEqual(len(crossed), 9)
        for field in ['field1', 'field2', 'field3', 'field4']:
            self.assertIn(field, crossed.columns)
        filtered = crossed[(crossed['field1'] == 1) & (crossed['field2'] == 2)]
        self.assertEqual(len(filtered), 3)

    def test_self_join_dataframe(self):
        df1, _ = get_sample_data()
        crossed = df_crossjoin(df1, df1, suffixes=['_left', '_right'])
        self.assertEqual(len(crossed), 9)
        for field in ['field1_left', 'field2_right']:
            self.assertIn(field, crossed.columns)
        filtered = crossed[(crossed['field1_left'] == 1) & (crossed['field2_left'] == 2)]
        self.assertEqual(len(filtered), 3)


class testMatcher(unittest.TestCase):

    def test_exact_field_match(self):
        df1, df2 = get_match_data()
        matcher = Matcher(df1, 'id1', df2, 'id2')
        matched = matcher.match_on_field(['field'])
        self.assertEqual(len(matched), 1)
        self.assertEqual(matched.iloc[0]['field'], 'baz')

    def test_function_match(self):
        df1, df2 = get_match_data()
        matcher = Matcher(df1, 'id1', df2, 'id2')
        matched = matcher.match_on_function(
            lambda row: row['field'][:3]
        )
        self.assertEqual(len(matched), 3)
        self.assertEqual(matched.iloc[0]['id1'], 1)
        self.assertEqual(matched.iloc[0]['id2'], 100)
        self.assertEqual(matched.iloc[1]['id1'], 2)
        self.assertEqual(matched.iloc[1]['id2'], 101)

    def test_levenshtein(self):
        df1, df2 = get_levenshtein_data()
        matcher = Matcher(df1, 'id1', df2, 'id2')
        matched = matcher.levenshtein([
                {'field_name': 'first', 'precision': 0},
                {'field_name': 'last', 'precision': 1}
        ])
        self.assertEqual(len(matched), 1)
        self.assertEqual(matched.iloc[0]['id1'], 3)
        self.assertEqual(matched.iloc[0]['id2'], 102)
        matched = matcher.levenshtein([
                {'field_name': 'first', 'precision': 2},
                {'field_name': 'last', 'precision': 1}
        ])
        self.assertEqual(len(matched), 2)
        self.assertEqual(matched.iloc[0]['id1'], 1)
        self.assertEqual(matched.iloc[0]['id2'], 100)

    def test_multiple_criteria(self):
        df1, df2 = get_levenshtein_data()
        match_types = get_match_types()
        matcher = Matcher(df1, 'id1', df2, 'id2')
        matched = matcher.create_matches(match_types)
        self.assertEqual(str(matched), "< MatchResult: 3 records; id1 to id2 >")
        # TODO write asserts


def get_sample_data():
    df1 = pd.DataFrame(
        [
            {'field1': 1, 'field2': 2},
            {'field1': 2, 'field2': 3},
            {'field1': 3, 'field2': 4}
        ]
    )
    df2 = pd.DataFrame(
        [
            {'field3': 1, 'field4': 2},
            {'field3': 2, 'field4': 3},
            {'field3': 3, 'field4': 4}
        ]
    )
    return df1, df2


def get_match_data():
    df1 = pd.DataFrame(
        [
            {'field': 'foo', 'id1': 1},
            {'field': 'bar', 'id1': 2},
            {'field': 'baz', 'id1': 3}
        ]
    )
    df2 = pd.DataFrame(
        [
            {'field': 'food', 'id2': 100},
            {'field': 'barn', 'id2': 101},
            {'field': 'baz', 'id2': 102}
        ]
    )
    return df1, df2


def get_levenshtein_data():
    df1 = pd.DataFrame(
        [
            {'id1': 1, 'first': 'Jack', 'last': 'Smith'},
            {'id1': 2, 'first': 'Jane', 'last': 'Jones'},
            {'id1': 3, 'first': 'Bob', 'last': 'Mitten'},
        ]
    )
    df2 = pd.DataFrame(
        [
            {'id2': 100, 'first': 'Jake', 'last': 'Smyth'},
            {'id2': 101, 'first': 'Joan', 'last': 'Jeans'},
            {'id2': 102, 'first': 'Bob', 'last': 'Kitten'},
        ]
    )
    return df1, df2


def get_match_types():
    return [
        {
            'type_id': 1,
            'method': 'exact_match',
            'fields': ['first']
        },
        {
            'type_id': 2,
            'method': 'levenshtein',
            'fields': [
                {'field_name': 'first', 'precision': 2},
                {'field_name': 'last', 'precision': 1}
            ]
        }
    ]
