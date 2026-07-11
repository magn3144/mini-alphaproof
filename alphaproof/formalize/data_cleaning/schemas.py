from typing import Any


MATH_WORD_SPLIT_PROBLEM_SCHEMA: dict[str, Any] = {
    'type': 'object',
    'properties': {
        'problems': {
            'type': 'array',
            'minItems': 1,
            'items': {
                'type': 'object',
                'properties': {
                    'problem': {'type': 'string'},
                    'answer': {'type': ['string', 'null']},
                },
                'required': ['problem', 'answer'],
                'additionalProperties': False,
            },
        },
    },
    'required': ['problems'],
    'additionalProperties': False,
}

MCQ_SPLIT_PROBLEM_SCHEMA: dict[str, Any] = {
    'type': 'object',
    'properties': {
        'problems': {
            'type': 'array',
            'minItems': 1,
            'items': {
                'type': 'object',
                'properties': {
                    'problem': {'type': 'string'},
                    'answer': {'type': 'string', 'enum': ['true', 'false']},
                },
                'required': ['problem', 'answer'],
                'additionalProperties': False,
            },
        },
    },
    'required': ['problems'],
    'additionalProperties': False,
}

PROOF_SPLIT_PROBLEM_SCHEMA: dict[str, Any] = {
    'type': 'object',
    'properties': {
        'problems': {
            'type': 'array',
            'minItems': 1,
            'items': {
                'type': 'object',
                'properties': {
                    'problem': {'type': 'string'},
                },
                'required': ['problem'],
                'additionalProperties': False,
            },
        },
    },
    'required': ['problems'],
    'additionalProperties': False,
}

REMOVE_OPTIONS_SCHEMA: dict[str, Any] = {
    'type': 'object',
    'properties': {
        'problem': {'type': 'string'},
        'answer': {'type': 'string'},
    },
    'required': ['problem', 'answer'],
    'additionalProperties': False,
}

ANSWER_PROOF_SCHEMA: dict[str, Any] = {
    'type': 'object',
    'properties': {
        'problem': {'type': 'string'},
    },
    'required': ['problem'],
    'additionalProperties': False,
}

ANSWERLESS_PROOF_SCHEMA: dict[str, Any] = {
    'type': 'object',
    'properties': {
        'problem': {'type': 'string'},
    },
    'required': ['problem'],
    'additionalProperties': False,
}
