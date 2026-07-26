from app.utils.ipa import to_ipa


def test_to_ipa_matches_known_pronunciation():
    assert to_ipa("rabbit", language="en") == "ɹæbɪt"
    assert to_ipa("cat", language="en") == "kæt"


def test_to_ipa_defaults_to_english():
    assert to_ipa("cat") == to_ipa("cat", language="en")


def test_to_ipa_differs_by_language():
    assert to_ipa("cat", language="en") != to_ipa("cat", language="hi")
