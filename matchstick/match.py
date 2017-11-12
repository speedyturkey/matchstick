from functools import reduce
from operator import and_
import pandas as pd
import Levenshtein


def crossjoin_dataframes(df1, df2, **kwargs):
    """
    Creates a cross-join (Cartesian product) between two DataFrames.
    Adapted from from mkonrad.net

    :param df1: First DataFrame.
    :param df2: Second DataFrame.
    :param kwargs: Any keyword arguments to be applied to the resulting DataFrame.
    :return: A DataFrame consisting of the cross-join between the provided DataFrames.
    """
    tmp1 = df1.copy()
    tmp2 = df2.copy()
    tmp1['__tmp'] = 1
    tmp2['__tmp'] = 1

    product = pd.merge(tmp1, tmp2, on='__tmp', **kwargs).drop('__tmp', axis=1)

    return product


def remove_duplicate_matches(data, id_fields):
    """

    :param data: A dataframe containing (at least) left_id_field, right_id_field, and match_type.
    :param id_fields: A list containing the names of the left_id_field and right_id_field.;
    :return: De-duped match data, keeping only the highest-priority match type.
    """

    assert isinstance(data, pd.DataFrame)
    assert isinstance(id_fields, list)
    for field in id_fields:
        assert field in data.columns

    return data.sort_values('match_type').drop_duplicates(
        subset=id_fields,
        keep='first')

class Matcher(object):
    """
    Matcher is used to perform matching or record linkage between two datasets (or between one dataset and itself).
    Provides a flexible mechanism to return results using specified match criteria.
    """
    def __init__(self, left_data, left_id_field, right_data, right_id_field, suffixes=None):
        if isinstance(left_data, pd.DataFrame):
            self.left_data = left_data
        else:
            self.left_data = pd.DataFrame(left_data)
        if left_id_field not in self.left_data.columns:
            raise ValueError("Field {} not present in left data.".format(left_id_field))
        self.left_id_field = left_id_field
        if isinstance(right_data, pd.DataFrame):
            self.right_data = right_data
        else:
            self.right_data = pd.DataFrame(right_data)
        if right_id_field not in self.right_data.columns:
            raise ValueError("Field {} not present in right data.".format(right_id_field))
        self.right_id_field = right_id_field
        self.suffixes = suffixes or ['_left', '_right']

    def __str__(self):
        return "< MatchMaker: {} records identified by {}; {} records identified by {} >".format(
            len(self.left_data),
            self.left_id_field,
            len(self.right_data),
            self.right_id_field
        )

    def create_matches(self, match_criteria):
        """
        Iterates through provided match_criteria, performing matches between the provided data sets.
        :param match_criteria: A dictionary following certain conventions - see documentation.
        :return: MatchResult object containing successful match data.
        """
        self.validate_match_criteria(match_criteria)
        results = []
        for match_type in match_criteria:
            type_id = match_type.get('type_id')
            if match_type['method'] == 'exact_match':
                match_type_result = self.match_on_field(match_type['fields'], type_id)
            elif match_type['method'] == 'function':
                match_type_result = self.match_on_function(match_type['function'], type_id)
            elif match_type['method'] == 'levenshtein':
                match_type_result = self.levenshtein(match_type['fields'], type_id)
            results.append(match_type_result)
        combined_results = pd.concat(results)
        return MatchResult(combined_results, self.left_id_field, self.right_id_field)

    def match_on_field(self, field_list, type_id=None):
        """
        Returns data which exactly matches on one or more provided fields.

        :param field_list: A List of one or more fields upon which to match exactly.
        :param type_id: Integer (optional) indicating the specific match type within a set of match criteria.
        :return: A DataFrame of the results matched using the provided parameters.
        """

        merged_df = pd.merge(self.left_data, self.right_data, on=field_list, suffixes=self.suffixes)
        merged_df['matched_to'] = merged_df[self.left_id_field]
        merged_df['match_type'] = type_id
        return merged_df

    def match_on_function(self, func, type_id=None):
        """
        Returns data matched using the provided function.

        :param func: A callable which will be applied to the provided datasets.
        :param type_id: Integer (optional) indicating the specific match type within a set of match criteria.
        :return: A DataFrame of the results matched using the provided parameters.
        """
        if hasattr(func, '__name__'):
            match_column = func.__name__
        else:
            match_column = 'MatchColumn'
        self.left_data[match_column] = self.left_data.apply(func, axis=1)
        self.right_data[match_column] = self.right_data.apply(func, axis=1)
        result = self.match_on_field([match_column], type_id)
        self.left_data.drop(match_column, axis=1, inplace=True)
        self.right_data.drop(match_column, axis=1, inplace=True)
        result.drop(match_column, axis=1, inplace=True)
        return result

    def levenshtein(self, fields, type_id=None):
        """
        Returns data matched using the Levenshtein Distance algorithm (difference between two strings as
        measured by the number of edits necessary to make them equal). A cross-join is performed between the two
        provided datasets, and the Levenshtein Distance is calculated for each applicable field. Results are filtered
        to the specified level of precision.

        :param fields: A list of dictionaries, including field names and desired precision. Precision is an integer
        indicating the maximum allowed Levenshtein distance for a record to be matched.
        :param type_id: Integer (optional) indicating the specific match type within a set of match criteria.
        :return: A DataFrame of the results matched using the provided parameters.
        """
        for field in fields:
            assert field['field_name'] in self.left_data.columns
            assert field['field_name'] in self.right_data.columns

        for field in fields:
            field['left'] = field['field_name'] + self.suffixes[0]
            field['right'] = field['field_name'] + self.suffixes[1]
            field['levenshtein'] = field['field_name'] + '_levenshtein_distance'

        crossed = crossjoin_dataframes(self.left_data, self.right_data, suffixes=self.suffixes)
        crossed['match_type'] = type_id

        # Minimum Levenshtein distance between two strings is the difference in their length.
        # Because calculating Levenshtein across the entire cross-joined data may be expensive,
        # we eliminate candidates where the difference in length makes a successful match impossible.
        for field in fields:
            crossed['length_diff'] = crossed.apply(
                lambda row: abs(len(row[field['left']]) - len(row[field['right']])),
                axis=1
            )
            crossed = crossed[crossed['length_diff'] <= field['precision']]

        for field in fields:
            crossed[field['levenshtein']] = crossed.apply(
                lambda row: Levenshtein.distance(row[field['left']], row[field['right']]),
                axis=1
            )
        filters = (crossed[field['levenshtein']].le(field['precision']) for field in fields)
        chained_filters = reduce(and_, filters)
        # Only return results inside the specified distance.
        crossed['matched_to'] = crossed[self.left_id_field]
        crossed = crossed[chained_filters]
        return crossed

    @staticmethod
    def validate_match_criteria(match_criteria):
        """
        Ensure that match_criteria are properly defined according to required logic.
        Malformed match_criteria will result in an assertion error.

        :param match_criteria: A list of dictionaries defining a set of match criteria. See documentation
        for the correct format and allowed parameters.
        """
        for match_type in match_criteria:
            assert 'method' in match_type.keys()
            assert match_type['method'] in ['exact_match', 'function', 'levenshtein']
            if match_type['method'] == 'exact_match':
                assert 'fields' in match_type.keys()
            if match_type['method'] == 'function':
                assert 'function' in match_type.keys()
                assert hasattr(match_type['function'], '__call__')
            if match_type['method'] == 'levenshtein':
                assert 'fields' in match_type.keys()
                for field in match_type['fields']:
                    assert 'field_name' in field.keys()
                    assert 'precision' in field.keys()
                    assert isinstance(field['precision'], int)


class MatchResult(object):
    """
    MatchResult is a container for data generated by Matcher.create_matches().
    """
    def __init__(self, matched_data, left_id_field, right_id_field):
        self.matched_data = matched_data
        self.left_id_field = left_id_field
        self.right_id_field = right_id_field
        self.core_fields = [left_id_field, right_id_field, 'match_type']
        self.core_data = matched_data[self.core_fields]

    def __str__(self):
        return "< MatchResult: {} records; {} to {} >".format(
            len(self.matched_data),
            self.left_id_field,
            self.right_id_field
        )

    @property
    def unique_matches(self):
        return remove_duplicate_matches(
            self.core_data,
            [self.left_id_field, self.right_id_field]
        )



