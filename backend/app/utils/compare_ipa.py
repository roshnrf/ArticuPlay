from dataclasses import dataclass, field

from app.utils.ipa import tokenize_ipa

_VOWELS = {
    "iː", "ɪ", "e", "ɛ", "æ", "ɑː", "ɒ", "ɔː", "ʊ", "uː", "ʌ", "ɜː", "ə", "ɐ",
    "eɪ", "aɪ", "ɔɪ", "aʊ", "oʊ", "ɪə", "eə", "ʊə", "əʊ",
}


@dataclass
class PhonemeError:
    position: int  # index into the target phoneme sequence
    expected: str | None
    got: str | None
    type: str  # substitution | omission | addition | cluster_reduction | syllable_deletion


@dataclass
class ComparisonResult:
    accuracy: float
    errors: list[PhonemeError]
    target_phonemes: list[str] = field(default_factory=list)
    child_phonemes: list[str] = field(default_factory=list)


def _align(target: list[str], child: list[str]) -> list[tuple[str, int | None, int | None]]:
    """Wagner-Fischer edit-distance alignment (same idea as a text diff, applied to
    phoneme sequences). Returns an ordered list of (op, target_index, child_index),
    op in {"match", "substitute", "delete", "insert"}; the unused index is None."""
    n, m = len(target), len(child)
    cost = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(1, n + 1):
        cost[i][0] = i
    for j in range(1, m + 1):
        cost[0][j] = j
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            if target[i - 1] == child[j - 1]:
                cost[i][j] = cost[i - 1][j - 1]
            else:
                cost[i][j] = 1 + min(cost[i - 1][j - 1], cost[i - 1][j], cost[i][j - 1])

    ops: list[tuple[str, int | None, int | None]] = []
    i, j = n, m
    while i > 0 or j > 0:
        if i > 0 and j > 0 and target[i - 1] == child[j - 1]:
            ops.append(("match", i - 1, j - 1))
            i, j = i - 1, j - 1
        elif i > 0 and j > 0 and cost[i][j] == cost[i - 1][j - 1] + 1:
            ops.append(("substitute", i - 1, j - 1))
            i, j = i - 1, j - 1
        elif i > 0 and cost[i][j] == cost[i - 1][j] + 1:
            ops.append(("delete", i - 1, None))
            i -= 1
        else:
            ops.append(("insert", None, j - 1))
            j -= 1
    ops.reverse()
    return ops


def _classify_omission_runs(errors: list[PhonemeError], target: list[str]) -> None:
    """Refines consecutive omissions in place: a run touching a vowel is a syllable
    deletion (a whole syllable dropped); a lone omission next to another consonant in
    the target is a cluster reduction (e.g. "star"->"tar" drops /s/ from the /st/ cluster).
    An isolated omission with vowels on both sides stays a plain omission."""
    omission_positions = [e.position for e in errors if e.type == "omission"]
    if not omission_positions:
        return

    runs: list[list[int]] = []
    for pos in sorted(omission_positions):
        if runs and pos == runs[-1][-1] + 1:
            runs[-1].append(pos)
        else:
            runs.append([pos])

    by_position = {e.position: e for e in errors if e.type == "omission"}
    for run in runs:
        touches_vowel = any(target[p] in _VOWELS for p in run)
        if touches_vowel:
            for p in run:
                by_position[p].type = "syllable_deletion"
        elif len(run) == 1:
            p = run[0]
            neighbor_positions = [p - 1, p + 1]
            has_consonant_neighbor = any(
                0 <= n < len(target) and target[n] not in _VOWELS for n in neighbor_positions
            )
            if has_consonant_neighbor:
                by_position[p].type = "cluster_reduction"


def compare_ipa(target_ipa: str, child_ipa: str) -> ComparisonResult:
    """Compares a target pronunciation against a child's attempt, phoneme by phoneme.
    Both inputs are raw espeak-ng IPA strings (e.g. from to_ipa())."""
    target = tokenize_ipa(target_ipa)
    child = tokenize_ipa(child_ipa)

    if not target:
        return ComparisonResult(accuracy=1.0, errors=[], target_phonemes=target, child_phonemes=child)

    ops = _align(target, child)
    errors: list[PhonemeError] = []
    for op, ti, ci in ops:
        if op == "match":
            continue
        elif op == "substitute":
            errors.append(PhonemeError(position=ti, expected=target[ti], got=child[ci], type="substitution"))
        elif op == "delete":
            errors.append(PhonemeError(position=ti, expected=target[ti], got=None, type="omission"))
        else:  # insert
            errors.append(PhonemeError(position=ti if ti is not None else -1, expected=None, got=child[ci], type="addition"))

    _classify_omission_runs(errors, target)

    accuracy = max(0.0, (len(target) - len(errors)) / len(target))
    return ComparisonResult(accuracy=accuracy, errors=errors, target_phonemes=target, child_phonemes=child)
