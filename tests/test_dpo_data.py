from alignment.dpo_data import parse_hh, parse_ultrafeedback


def test_hh_single_turn_pair():
    chosen = "\n\nHuman: How do I boil an egg?\n\nAssistant: Put it in boiling water for 9 minutes."
    rejected = "\n\nHuman: How do I boil an egg?\n\nAssistant: No idea."
    assert parse_hh(chosen, rejected) == (
        "How do I boil an egg?", "Put it in boiling water for 9 minutes.", "No idea.")


def test_hh_multi_turn_and_mismatch_rejected():
    multi = "\n\nHuman: hi\n\nAssistant: hello\n\nHuman: bye\n\nAssistant: bye"
    single = "\n\nHuman: hi\n\nAssistant: hello"
    assert parse_hh(multi, multi.replace("bye", "ciao")) is None
    assert parse_hh(single, "\n\nHuman: other\n\nAssistant: hello there") is None
    assert parse_hh(single, single) is None  # identical responses carry no preference


def test_ultrafeedback_pair_and_ties():
    row = {"prompt": "Name a prime.",
           "chosen": [{"role": "user", "content": "Name a prime."}, {"role": "assistant", "content": "7"}],
           "rejected": [{"role": "user", "content": "Name a prime."}, {"role": "assistant", "content": "8"}],
           "score_chosen": 9.0, "score_rejected": 2.0}
    assert parse_ultrafeedback(row) == ("Name a prime.", "7", "8")
    assert parse_ultrafeedback({**row, "score_rejected": 9.0}) is None
