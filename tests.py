import unittest
import numpy as np
import pandas as pd
from matchstick import Matcher
from matchstick import crossjoin_dataframes, remove_duplicate_matches


class TestFunctions(unittest.TestCase):

    def test_join_two_dataframes(self):
        df1, df2 = get_sample_data()
        crossed = crossjoin_dataframes(df1, df2)
        self.assertEqual(len(crossed), 9)
        for field in ['field1', 'field2', 'field3', 'field4']:
            self.assertIn(field, crossed.columns)
        filtered = crossed[(crossed['field1'] == 1) & (crossed['field2'] == 2)]
        self.assertEqual(len(filtered), 3)

    def test_self_join_dataframe(self):
        df1, _ = get_sample_data()
        crossed = crossjoin_dataframes(df1, df1, suffixes=['_left', '_right'])
        self.assertEqual(len(crossed), 9)
        for field in ['field1_left', 'field2_right']:
            self.assertIn(field, crossed.columns)
        filtered = crossed[(crossed['field1_left'] == 1) & (crossed['field2_left'] == 2)]
        self.assertEqual(len(filtered), 3)

    def test_remove_duplicate_matches(self):
        df = pd.DataFrame([
            {'id1': 1, 'id2': 1, 'match_type': 1},
            {'id1': 1, 'id2': 1, 'match_type': 2},
            {'id1': 2, 'id2': 2, 'match_type': 2},
            {'id1': 2, 'id2': 2, 'match_type': 3},
            {'id1': 2, 'id2': 2, 'match_type': 4},
            {'id1': 3, 'id2': 3, 'match_type': None},
            {'id1': 3, 'id2': 3, 'match_type': None},
        ])
        unique = remove_duplicate_matches(df, ['id1', 'id2'])
        self.assertEqual(len(unique[(unique['id1'] == 1) & (unique['id2'] == 1)]), 1)
        self.assertEqual(unique.iloc[0]['match_type'], 1)
        self.assertEqual(unique.iloc[1]['match_type'], 2)
        self.assertTrue(np.isnan(unique.iloc[2]['match_type']))


class TestMatcher(unittest.TestCase):

    def test_validate_match_criteria(self):
        no_method = [{
            'type_id': 1,
            'fields': ['first_name', 'last_name']
        }]
        invalid_method = [{
            'type_id': 1,
            'method': 'match_exactly',
            'fields': ['first_name', 'last_name']
        }]
        exact_missing_fields = [{
            'type_id': 1,
            'method': 'exact_match',
        }]
        missing_function = [{
            'type_id': 1,
            'method': 'function',
        }]
        no_callable = [{
            'type_id': 1,
            'method': 'function',
            'function': "You can't call a string!"
        }]
        levenshtein_missing_fields = [{
            'type_id': 1,
            'method': 'levenshtein',
        }]
        levenshtein_missing_field_name = [{
            'type_id': 4,
            'method': 'levenshtein',
            'fields': [
                {'field_name': 'first_name', 'precision': 1},
                {'precision': 2}
            ]
        }]
        levenshtein_missing_precision = [{
            'type_id': 4,
            'method': 'levenshtein',
            'fields': [
                {'field_name': 'first_name'},
            ]
        }]
        levenshtein_invalid_precision = [{
            'type_id': 4,
            'method': 'levenshtein',
            'fields': [
                {'field_name': 'first_name', 'precision': '1'},
            ]
        }]
        for bad_match_criteria in [
            no_method,
            invalid_method,
            exact_missing_fields,
            missing_function,
            no_callable,
            levenshtein_missing_fields,
            levenshtein_missing_field_name,
            levenshtein_missing_precision,
            levenshtein_invalid_precision
        ]:
            with self.assertRaises(AssertionError):
                Matcher.validate_match_criteria(bad_match_criteria)

    def test_constructor(self):
        list1, list2 = get_lists_of_lists()
        matcher = Matcher(list1, 'id1', list2, 'id2')
        self.assertIsInstance(matcher.left_data, pd.DataFrame)
        self.assertIsInstance(matcher.right_data, pd.DataFrame)
        with self.assertRaises(ValueError):
            Matcher(list1, 'id2', list2, 'id2')
        with self.assertRaises(ValueError):
            Matcher(list1, 'id1', list2, 'id1')

    def test_exact_field_match(self):
        df1, df2 = get_match_data()
        matcher = Matcher(df1, 'id1', df2, 'id2')
        self.assertIsInstance(matcher.left_data, pd.DataFrame)
        self.assertIsInstance(matcher.right_data, pd.DataFrame)
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
        self.assertEqual(str(matcher), "< MatchMaker: 3 records identified by id1; 3 records identified by id2 >")
        self.assertEqual(str(matched), "< MatchResult: 5 records; id1 to id2 >")
        self.assertEqual(len(matched.matched_data[matched.matched_data['match_type'] == 1]), 1)
        self.assertEqual(len(matched.matched_data[matched.matched_data['match_type'] == 2]), 2)
        self.assertEqual(len(matched.matched_data[matched.matched_data['match_type'] == 3]), 2)
        self.assertEqual(len(matched.matched_data[(matched.matched_data['id1'] == 1) & (matched.matched_data['id2'] == 100)]), 2)
        self.assertEqual(len(matched.matched_data[(matched.matched_data['id1'] == 2) & (matched.matched_data['id2'] == 101)]), 1)
        self.assertEqual(len(matched.matched_data[(matched.matched_data['id1'] == 3) & (matched.matched_data['id2'] == 102)]), 2)
        self.assertEqual(len(matched.unique_matches), len(matched.matched_data) - 2)
        self.assertEqual(matched.unique_matches.iloc[0]['match_type'], 1)
        self.assertEqual(matched.unique_matches.iloc[1]['match_type'], 2)
        self.assertEqual(matched.unique_matches.iloc[2]['match_type'], 3)


def get_lists_of_lists():
    list1 = [
        {'field': 'foo', 'id1': 1},
        {'field': 'bar', 'id1': 2},
        {'field': 'baz', 'id1': 3}
    ]
    list2 = [
        {'field': 'food', 'id2': 100},
        {'field': 'barn', 'id2': 101},
        {'field': 'baz', 'id2': 102}
    ]

    return list1, list2


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
            'type_id': 3,
            'method': 'function',
            'function': lambda row: row['first'][:1] + row['last'][:1]
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
