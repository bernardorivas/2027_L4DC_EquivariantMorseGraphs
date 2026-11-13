"""Check the shared numerical environment and physical systems."""

import json

from common.workflow import require_environment


def main():
    require_environment()
    from common.check_environment import check
    from common.check_systems import check as check_systems
    report = {"environment": check(), "systems": check_systems(with_emlp=True)}
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
