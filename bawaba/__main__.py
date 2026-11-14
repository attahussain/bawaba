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
    args = sys.argv[1:]
    photo = None
    if "--photo" in args:  # python -m bawaba "message" --photo path/to/photo.jpg
        i = args.index("--photo")
        photo, args = args[i + 1], args[:i] + args[i + 2:]
    messages = [" ".join(args)] if args else iter(lambda: input("tenant> "), "")
    for message in messages:
        out = desk.handle(message, photo=photo)
        print(out.reply)
        print(json.dumps({k: v for k, v in out.to_dict().items() if k != "reply"}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
