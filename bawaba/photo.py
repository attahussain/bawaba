"""Photo triage: how urgent is the problem in a tenant's photo, and what kind of problem is it?"""

from dataclasses import dataclass

PHOTO_QUESTIONS = {
    "urgency": {"type": "choice", "instructions": "How urgent is the problem shown in this photo of an apartment?",
                "criteria": {"routine": "no problem, or a minor cosmetic issue",
                             "soon": "damage that should be repaired within a day or two",
                             "emergency": "safety risk or active damage: collapse, flooding, fire, exposed electrics"}},
    "issue": {"type": "choice", "instructions": "What kind of problem does the photo show?",
              "criteria": {"water": "leak, flooding, water stains or water damage",
                           "mould": "mould or damp patches",
                           "broken": "broken glass, fixtures or fittings",
                           "structural": "collapsed or holed ceiling, walls or floor",
                           "minor": "small cosmetic defect such as a crack in grout or caulk",
                           "none": "no visible problem"}},
}
SEVERITY = {"routine": 0, "soon": 1, "emergency": 2}


@dataclass
class PhotoTriage:
    urgency: str
    issue: str
    confidence: float | None


def triage_photo(decider, path: str, message: str = "") -> PhotoTriage:
    answers = decider.predict_image(path, PHOTO_QUESTIONS, text=message)
    conf = min((max(a["probabilities"].values()) for a in answers.values() if a.get("probabilities")), default=None)
    return PhotoTriage(answers["urgency"]["choice"], answers["issue"]["choice"], conf)


def most_urgent(*levels: str) -> str:
    return max(levels, key=lambda u: SEVERITY.get(u, 0))
