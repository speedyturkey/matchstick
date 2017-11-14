population_data = [
    {'unique_id': 100, 'first_name': 'Mickey', 'last_name': 'Mouse', 'ssn': 123},
    {'unique_id': 101, 'first_name': 'Daffy', 'last_name': 'Duck', 'ssn': 456},
    {'unique_id': 102, 'first_name': 'Bodoni', 'last_name': 'Cat', 'ssn': 789},
    {'unique_id': 103, 'first_name': 'Pip', 'last_name': 'Cat', 'ssn': 345},
]

new_data = [
    {'new_id': 1, 'first_name': 'Mikkey', 'last_name': 'Mouse', 'ssn': 123},
    {'new_id': 2, 'first_name': 'Bodni', 'last_name': 'Cat', 'ssn': 789},
    {'new_id': 3, 'first_name': 'Pip', 'last_name': 'Cat', 'ssn': 345},
    {'new_id': 4, 'first_name': 'Mick', 'last_name': 'Mouse', 'ssn': 123},
    {'new_id': 5, 'first_name': 'Mick', 'last_name': 'Moose', 'ssn': 123},
    {'new_id': 6, 'first_name': 'Joe', 'last_name': 'Smith', 'ssn': 999}
]

sample_result = [
    {'unique_id': 1, 'new_id': 1, 'match_type': 1},
    {'unique_id': 1, 'new_id': 1, 'match_type': 2},
    {'unique_id': 2, 'new_id': 4, 'match_type': 2},
    {'unique_id': 2, 'new_id': 4, 'match_type': 3},
    {'unique_id': 3, 'new_id': 6, 'match_type': 1},
    {'unique_id': 3, 'new_id': 6, 'match_type': 3},
]