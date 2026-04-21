from __future__ import annotations

import json
import os
import unittest

from xdof_sim.config import get_i2rt_sim_config
from xdof_sim.task_specs import get_task_spec, maybe_get_task_spec


class TaskSpecTests(unittest.TestCase):
    def test_resolves_bottles_alias_and_prompt(self) -> None:
        by_alias = get_task_spec("bottles")
        by_prompt = get_task_spec("throw plastic bottles in bin")
        self.assertEqual(by_alias, by_prompt)
        self.assertEqual(by_alias.env_task, "bottles")
        self.assertEqual(by_alias.name, "throw_plastic_bottles_in_bin")
        self.assertEqual(by_alias.evaluator_name, "bottles_in_bin")
        self.assertEqual(by_alias.evaluator_kwargs()["success_count"], 2)

    def test_resolves_spell_alias(self) -> None:
        spec = get_task_spec("sim_spell_cat")
        self.assertEqual(spec.name, "spell_cat")
        self.assertEqual(spec.env_task, "blocks")
        self.assertEqual(spec.prompt, "spell cat")

    def test_resolves_new_20260414_delivery_aliases(self) -> None:
        pouring = get_task_spec("sim_pouring_beads")
        bottles = get_task_spec("sim_put the plastic bottles in the bin")
        self.assertEqual(pouring.name, "pouring")
        self.assertEqual(pouring.env_task, "pour")
        self.assertEqual(bottles.name, "throw_plastic_bottles_in_bin")
        self.assertEqual(bottles.env_task, "bottles")

    def test_maybe_get_task_spec_returns_none_for_unknown(self) -> None:
        self.assertIsNone(maybe_get_task_spec("definitely_not_a_real_task"))


class SimConfigOverrideTests(unittest.TestCase):
    def test_deploy_init_q_overrides_sim_config(self) -> None:
        old = os.environ.get("DEPLOY_INIT_Q")
        try:
            override = list(range(14))
            os.environ["DEPLOY_INIT_Q"] = json.dumps(override)
            config = get_i2rt_sim_config()
        finally:
            if old is None:
                os.environ.pop("DEPLOY_INIT_Q", None)
            else:
                os.environ["DEPLOY_INIT_Q"] = old

        self.assertEqual(config.robots["left"].init_q, override[:7])
        self.assertEqual(config.robots["right"].init_q, override[7:])


if __name__ == "__main__":
    unittest.main()
