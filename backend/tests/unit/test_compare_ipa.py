from app.utils.compare_ipa import compare_ipa
from app.utils.ipa import to_ipa


def test_exact_match_has_full_accuracy_and_no_errors():
    result = compare_ipa("ɹæbɪt", "ɹæbɪt")
    assert result.accuracy == 1.0
    assert result.errors == []


def test_substitution_detected():
    result = compare_ipa("kæt", "bæt")
    assert len(result.errors) == 1
    error = result.errors[0]
    assert error.type == "substitution"
    assert error.expected == "k"
    assert error.got == "b"
    assert error.position == 0


def test_plain_omission_when_not_adjacent_to_a_consonant():
    # "cat" said as "at" - drops /k/, but /k/'s only neighbor (/æ/) is a vowel,
    # so this isn't a cluster reduction, just a plain omission.
    result = compare_ipa("kæt", "æt")
    assert len(result.errors) == 1
    assert result.errors[0].type == "omission"
    assert result.errors[0].expected == "k"


def test_addition_detected():
    result = compare_ipa("kæt", "skæt")
    assert len(result.errors) == 1
    assert result.errors[0].type == "addition"
    assert result.errors[0].got == "s"


def test_cluster_reduction_matches_spec_example():
    # spoon -> poon, from the spec doc's own worked example
    target_ipa = to_ipa("spoon")
    child_ipa = to_ipa("poon")
    result = compare_ipa(target_ipa, child_ipa)
    assert len(result.errors) == 1
    assert result.errors[0].type == "cluster_reduction"
    assert result.errors[0].expected == "s"


def test_syllable_deletion_matches_spec_example():
    # banana -> nana, from the spec doc's own worked example
    result = compare_ipa("bənænə", "nænə")
    types = [e.type for e in result.errors]
    assert "syllable_deletion" in types
    deleted = [e for e in result.errors if e.type == "syllable_deletion"]
    assert {e.expected for e in deleted} == {"b", "ə"}


def test_accuracy_reflects_error_count_against_phoneme_total():
    # 5 phonemes, 1 substitution -> 4/5
    result = compare_ipa("ɹæbɪt", "wæbɪt")
    assert result.accuracy == 0.8


def test_accuracy_never_goes_negative_on_heavily_garbled_attempt():
    result = compare_ipa("kæt", "zzzzzzzzzz")
    assert result.accuracy >= 0.0


def test_empty_target_is_full_accuracy_by_definition():
    result = compare_ipa("", "kæt")
    assert result.accuracy == 1.0
