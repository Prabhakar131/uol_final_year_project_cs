"""The security model can run from a quantized GGUF file through llama.cpp.

No model weights are loaded; llama.cpp and the tokenizer are replaced by fakes.
"""
import os
import unittest
from unittest.mock import patch

from cloudir.ai_models import security_model


class FakeTokenizer:
    def apply_chat_template(self, messages, tokenize, add_generation_prompt):
        return "<|user|>\n" + messages[-1]["content"] + "\n<|assistant|>\n"

    def __call__(self, text, return_tensors=None, add_special_tokens=True):
        return {"input_ids": [128000] * add_special_tokens + [len(word) for word in text.split()]}


class FakeLlama:
    def __init__(self):
        self.calls = []
        self.closed = False

    def create_completion(self, prompt, **kwargs):
        self.calls.append((prompt, kwargs))
        return {"choices": [{"text": '{"verdict": "Weak Support"}'}]}

    def close(self):
        self.closed = True


class SecurityBackendTests(unittest.TestCase):
    def tearDown(self):
        security_model._gguf_model = None
        security_model._load_security_model.cache_clear()

    def test_llama_cpp_is_the_default_backend(self):
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop(security_model.SECURITY_BACKEND_ENV, None)
            self.assertEqual(security_model.security_backend(), "llama_cpp")
        with patch.dict(os.environ, {security_model.SECURITY_BACKEND_ENV: "Transformers"}):
            self.assertEqual(security_model.security_backend(), "transformers")
        with patch.dict(os.environ, {security_model.SECURITY_BACKEND_ENV: "mlx"}):
            with self.assertRaises(RuntimeError):
                security_model.security_backend()

    def test_llama_cpp_gets_template_token_ids_once_and_greedy_decoding(self):
        tokenizer, llama = FakeTokenizer(), FakeLlama()
        messages = [{"role": "user", "content": "judge this evidence"}]
        text = security_model._generate_security_response(tokenizer, llama, "mps", messages, 600)

        self.assertEqual(text, '{"verdict": "Weak Support"}')
        prompt, kwargs = llama.calls[0]
        rendered = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        # The rendered template carries its own begin-of-text; none is added again.
        self.assertEqual(prompt, tokenizer(rendered, add_special_tokens=False)["input_ids"])
        self.assertNotIn(128000, prompt)
        self.assertEqual(kwargs["max_tokens"], 600)
        self.assertEqual((kwargs["temperature"], kwargs["top_k"], kwargs["repeat_penalty"]), (0.0, 1, 1.0))

    def test_judgements_are_greedy_and_only_authoring_retries_sample(self):
        tokenizer, llama = FakeTokenizer(), FakeLlama()
        messages = [{"role": "user", "content": "write a timeline"}]
        security_model._generate_security_response(tokenizer, llama, "mps", messages, 100)
        security_model._generate_security_response(tokenizer, llama, "mps", messages, 100, sampling_seed=2)

        greedy, sampled = (kwargs for _, kwargs in llama.calls)
        self.assertEqual((greedy["temperature"], greedy["top_k"], greedy["seed"]), (0.0, 1, None))
        self.assertEqual((sampled["temperature"], sampled["seed"]), (0.5, 2))

    def test_backend_selects_gguf_loader_and_unload_closes_it(self):
        llama = FakeLlama()

        def fake_gguf_loader():
            security_model._gguf_model = llama
            return llama

        with patch.dict(os.environ, {"HF_SECURITY_MODEL_ID": "fdtn-ai/Foundation-Sec-8B-Instruct",
                                     security_model.SECURITY_BACKEND_ENV: "llama_cpp"}), \
             patch.object(security_model.AutoTokenizer, "from_pretrained", return_value=FakeTokenizer()), \
             patch.object(security_model, "_load_gguf_security_model", side_effect=fake_gguf_loader), \
             patch.object(security_model.AutoModelForCausalLM, "from_pretrained") as full_weights:
            model, _ = security_model._load_security_model()
            security_model.unload_security_model()

        self.assertIs(model, llama)
        full_weights.assert_not_called()
        self.assertTrue(llama.closed)
        self.assertIsNone(security_model._gguf_model)


class ThreatModelScenarioTypeTests(unittest.TestCase):
    def test_the_threat_model_names_the_scenario_being_prepared(self):
        # The prompt and output were fixed to "identity-management" for every scenario.
        import json

        prompts = []

        def fake_generate(**kwargs):
            prompts.append(kwargs["messages"][-1]["content"])
            return json.dumps({"scenario_type": "identity-management", "primary_risk": "Runaway spend",
                               "affected_assets": [], "likely_attack_path": [], "security_signals": [],
                               "investigation_goals": [], "recommended_evidence_sources": [],
                               "hidden_truth_candidates": []})

        with patch.object(security_model, "_load_security_model", return_value=(object(), object())), \
             patch.object(security_model, "_device", return_value="cpu"), \
             patch.object(security_model, "_generate_security_response", side_effect=fake_generate):
            result = security_model.normalise_acse_threat_model({}, {}, scenario_id="cost_management")

        self.assertEqual(result["scenario_type"], "cost-management")
        self.assertIn('ACSE-Eval "cost-management" scenario', prompts[0])
        self.assertNotIn("identity-management", prompts[0])


class TimelineExampleNameTests(unittest.TestCase):
    def test_each_supported_scenario_gets_its_own_example_names(self):
        # The timeline author copies the prompt's example names, so a shared
        # example gave every scenario the same suspect.
        from cloudir.dataset_preparation.select_case import TRAINING_SCENARIO_ALLOWLIST

        names = [security_model.example_names(s) for s in sorted(TRAINING_SCENARIO_ALLOWLIST)]
        for part in range(3):
            self.assertEqual(len({n.split(", ")[part] for n in names}), len(names))
        self.assertEqual(security_model.example_names("cost_management"), security_model.example_names("cost-management"))


if __name__ == "__main__":
    unittest.main()
