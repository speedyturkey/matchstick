import pandas as pd
from sample_data import population_data, new_data
from matchstick import Matcher

pd.set_option('display.width', 160)

match_types = [
    {
        'type_id': 1,
        'method': 'exact_match',
        'fields': ['first_name', 'last_name']
    },
    {
        'type_id': 3,
        'method': 'function',
        'function': lambda row: row['first_name'][:1] + row['last_name'][:1]
    },
    {
        'type_id': 4,
        'method': 'levenshtein',
        'fields': [
            {'field_name': 'first_name', 'precision': 1},
            {'field_name': 'last_name', 'precision': 2}
        ]
    },
]

matcher = Matcher(population_data, 'unique_id', new_data, 'new_id')
results = matcher.create_matches(match_types)
print("Matched data:")
print(results.matched_data)
print("Unique matches:")
print(results.unique_matches)
print("Unmatched records:")
print(matcher.unmatched(results.matched_data))
