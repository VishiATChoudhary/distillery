import re


class LLMJudge:
    """Grades the student's answer against the teacher's reference, 0..1. Reference-guided scoring
    with an explicit "ignore length" rule: LLM judges favour longer answers (Zheng et al. 2023)."""

    PROMPT = (
        "Rate how well the STUDENT answer matches the REFERENCE answer in correctness, content and style.\n"
        "Ignore length: a shorter answer that says the same thing is just as good.\n"
        "Reply with a single integer from 0 to 10 and nothing else.\n\n"
        "QUESTION:\n{q}\n\nREFERENCE:\n{ref}\n\nSTUDENT:\n{ans}"
    )

    def __init__(self, teacher):
        self.teacher = teacher

    def score(self, question, reference, answer):
        out = self.teacher.ask(self.PROMPT.format(q=question, ref=reference, ans=answer), max_tokens=8)
        m = re.search(r"\d+", out)
        return min(int(m.group()), 10) / 10 if m else 0.0


class OverlapJudge:
    """Offline judge: token F1 between student and reference."""

    def score(self, question, reference, answer):
        ref, ans = reference.lower().split(), answer.lower().split()
        common = sum(min(ref.count(w), ans.count(w)) for w in set(ans))
        if not common:
            return 0.0
        p, r = common / len(ans), common / len(ref)
        return 2 * p * r / (p + r)
