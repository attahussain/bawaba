"""Command line: `python -m bawaba "message"` or `python -m bawaba` for an interactive desk."""

import json
import sys

from bawaba.config import get_settings
from bawaba.kb import KnowledgeBase
from bawaba.pipeline import Desk


def main() -> int:
    settings = get_settings()
    kb = KnowledgeBase(settings)
    kb.ingest()
    desk = Desk(settings, kb)
    messages = [" ".join(sys.argv[1:])] if len(sys.argv) > 1 else iter(lambda: input("tenant> "), "")
    for message in messages:
        out = desk.handle(message)
        print(out.reply)
        print(json.dumps({k: v for k, v in out.to_dict().items() if k != "reply"}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
