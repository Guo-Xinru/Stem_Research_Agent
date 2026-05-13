from stem_research.prepare_qasper_mini import _collect_records, _iter_qas


def test_iter_qas_handles_qasper_dict_of_lists_shape() -> None:
    qas = {
        "question": ["What is the seed lexicon?"],
        "question_id": ["qid-1"],
        "answers": [
            {
                "answer": [
                    {
                        "unanswerable": False,
                        "extractive_spans": [],
                        "yes_no": None,
                        "free_form_answer": "a vocabulary of positive and negative predicates",
                        "evidence": ["The seed lexicon consists of positive and negative predicates."],
                    }
                ],
                "annotation_id": ["ann-1"],
            }
        ],
    }

    normalized = _iter_qas(qas)

    assert normalized == [
        {
            "question": "What is the seed lexicon?",
            "question_id": "qid-1",
            "answers": qas["answers"][0],
        }
    ]


def test_collect_records_handles_observed_qasper_parquet_shape() -> None:
    paper = {
        "id": "paper-1",
        "title": "Affective Events",
        "abstract": "This paper proposes a method.",
        "full_text": {
            "section_name": ["Proposed Method"],
            "paragraphs": [
                [
                    "The seed lexicon consists of positive and negative predicates. "
                    "If the predicate of an extracted event is in the seed lexicon, "
                    "the method assigns a polarity score to the event and propagates "
                    "labels through discourse relations in the raw corpus."
                ]
            ],
        },
        "qas": {
            "question": ["What is the seed lexicon?"],
            "question_id": ["qid-1"],
            "nlp_background": ["two"],
            "answers": [
                {
                    "answer": [
                        {
                            "unanswerable": False,
                            "extractive_spans": [],
                            "yes_no": None,
                            "free_form_answer": "a vocabulary of positive and negative predicates",
                            "evidence": [
                                "The seed lexicon consists of positive and negative predicates."
                            ],
                            "highlighted_evidence": [],
                        }
                    ],
                    "annotation_id": ["ann-1"],
                    "worker_id": ["worker-1"],
                }
            ],
        },
    }

    records = _collect_records([paper], split_name="train")

    assert len(records) == 1
    assert records[0]["id"] == "qid-1"
    assert records[0]["question"] == "What is the seed lexicon?"
    assert records[0]["reference_answer"] == "a vocabulary of positive and negative predicates"
    assert records[0]["evidence"][0]["section_name"] == "Proposed Method"
