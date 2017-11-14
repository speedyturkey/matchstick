from functools import reduce
from operator import and_
import pandas as pd
import Levenshtein


def crossjoin_dataframes(df1, df2, **kwargs):
    """
    Creates a cross-join (Cartesian product) between two DataFrames.

    Parameters
    ----------
    df1: DataFrame
        First (left) DataFrame
    df2 : DataFrame
        Second (right) DataFrame
    kwargs : dictionary
        Keyword arguments to be applied to the resulting DataFrame.

    Returns
    -------
    DataFrame : Cartesian product of df1 and df2
    """
    tmp1 = df1.copy()
    tmp2 = df2.copy()
    tmp1['__tmp'] = 1
    tmp2['__tmp'] = 1

    product = pd.merge(tmp1, tmp2, on='__tmp', **kwargs).drop('__tmp', axis=1)

    return product


def remove_duplicate_matches(data, id_fields):
    """
    Removes duplicate match data, keeping only the highest-priority match type per matched pair.

    Parameters
    ----------
    data : DataFrame
        Must contain (at least) columns for left_id_field, right_id_field, and match_type
    id_fields: list of strings
        Contains names of left_id_field and right_id_field
    Returns
    -------
    DataFrame : Containing distinct matches
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
    Provides a flexible mechanism to return results using customizable match criteria.

    Parameters
    ----------
    left_data : DataFrame or list of dictionaries
        Dataset that is being matched against (your "population" data)
    left_id_field : string
        Name of the field which uniquely identifies records within left_data
    right_data : DataFrame or list of dictionaries
        Dataset that is being matched (your "new" data)
    right_id_field : string
        Name of the field which uniquely identifies records within right_data
    :param suffixes: list of two strings, default ["_left", "_right"]
        Names applied to overlapping column names in left_data and right_data, respectively

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
        Iterates through provided match criteria, linking records between left_data and right_data.

        Parameters
        ----------
        match_criteria : list of dictionaries
            Each inner dictionary defines an individual match type, choosing from amongst several available
            mechanisms - exact match, apply function, or Levenshtein distance.

        Returns
        -------
        MatchResult object
            Contains information about the matches which were made between the two datasets.

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
        # Keep only the most important fields, then merge back original data.
        # Given variety of possible match types and column name permutations, it is easier to
        # ignore intermediate fields generated during the match process itself. By showing the original data
        # and match type, sufficient information on the nature of each match should be available.
        key_fields = [self.left_id_field, self.right_id_field, 'match_type']
        match_results = combined_results[key_fields]
        match_results = match_results.merge(self.left_data, on=self.left_id_field)
        match_results = match_results.merge(self.right_data, on=self.right_id_field)

        return MatchResult(match_results, self.left_id_field, self.right_id_field)

    def match_on_field(self, field_list, type_id=None):
        """
        Returns data which exactly matches on one or more provided fields.

        Parameters
        ----------
        field_list : list of one or more strings
            Records with matching values in each field across the two datasets are considered to be matches.
        type_id : int or string (optional), default None
            Used to uniquely identify an individual match type
        Returns
        -------
        DataFrame
        """
        merged_df = pd.merge(self.left_data, self.right_data, on=field_list, suffixes=self.suffixes)
        merged_df['matched_to'] = merged_df[self.left_id_field]
        merged_df['match_type'] = type_id
        return merged_df

    def match_on_function(self, func, type_id=None):
        """
        Returns data which matches after the provided callable is applied.

        Parameters
        ----------
        func : function, lambda, or other callable
            Will be applied to left_data and right_data
        type_id : int or string (optional), default None
            Used to uniquely identify an individual match type
        Returns
        -------
        DataFrame
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

        Parameters
        ----------
        fields : list of dictionaries
            including field names and desired precision eg {'field_name': first_name, 'precision': 2}
            Precision is an integer indicating the maximum allowed Levenshtein distance for a record to be matched.
        type_id : int or string (optional), default None
            Used to uniquely identify an individual match type
        Returns
        -------
        DataFrame : Contains columns from left_data and right_data where a match is made using provided criteria
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

        Parameters
        ----------
        match_criteria : list of dictionaries
            Each inner dictionary defines an individual match type, choosing from amongst several available
            mechanisms - exact match, apply function, or Levenshtein distance.

        Example Format
        --------------
        match_types = [
            {
                'type_id': 1,
                'method': 'exact_match',
                'fields': ['first_name', 'last_name']
            },
            {
                'type_id': 2,
                'method': 'function',
                'function': lambda row: row['first_name'][:1] + row['last_name'][:1]
            },
            {
                'type_id': 3,
                'method': 'levenshtein',
                'fields': [
                    {'field_name': 'first_name', 'precision': 1},
                    {'field_name': 'last_name', 'precision': 2}
                ]
            },
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

    def unmatched(self, match_results):
        """
        Identifies population of records within right_data where no match was made to a record in left_data.

        Parameters
        ----------
        match_results : DataFrame
            Contains full set of results made according to provided match criteria

        Returns
        -------
        Dataframe : Contains rows and columns from right_data where no match was made to left_data.
        """
        merged = pd.merge(match_results, self.right_data, on=self.right_id_field, how='right', indicator=True)
        right_only = merged[merged['_merge'] == 'right_only']
        return pd.merge(self.right_data, right_only)[self.right_data.columns]

    def match_to_multiple(self):
        """
        Used to check whether any records in right_data match to more than one record in left_data. Such an occurrence
        would suggest that one of the matches is made in error, or that records within left_data could potentially be
        combined.
        Returns
        -------
        """
        raise NotImplementedError


class MatchResult(object):
    """
    MatchResult is a container for data generated by Matcher.create_matches().

    Parameters
    ----------
    matched_data : DataFrame
        Contains records which have been matched to each other, including both unique IDs, match type, and original
        data columns
    left_id_field : string
        Name of the field which uniquely identifies records within left_data
    right_id_field : string
        Name of the field which uniquely identifies records within rogjt_data
    """
    def __init__(self, matched_data, left_id_field, right_id_field):
        self.matched_data = matched_data
        self.left_id_field = left_id_field
        self.right_id_field = right_id_field

    def __str__(self):
        return "< MatchResult: {} records; {} to {} >".format(
            len(self.matched_data),
            self.left_id_field,
            self.right_id_field
        )

    @property
    def unique_matches(self):
        """
        Returns
        -------
        DataFrame: Distinct combinations of matches made between left and right datasets, including only the "highest"
            match type per distinct pair of records.
        """
        return remove_duplicate_matches(
            self.matched_data,
            [self.left_id_field, self.right_id_field]
        )



