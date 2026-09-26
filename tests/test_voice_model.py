"""Whisper is told the answer is English, gets the AWS names, and silence is not an answer."""

import unittest
from types import SimpleNamespace

import torch

from cloudir.ai_models.voice_model import AWS_VOCABULARY, transcript_text, whisper_generate_kwargs


class FakeTokenizer:
    def __init__(self):
        self.prompts = []

    def get_prompt_ids(self, text, return_tensors=None):
        self.prompts.append(text)
        return torch.tensor([1, 2, 3])


def fake_pipeline(multilingual):
    config = SimpleNamespace(is_multilingual=multilingual)
    return SimpleNamespace(
        model=SimpleNamespace(generation_config=config, device=torch.device("cpu")),
        tokenizer=FakeTokenizer(),
    )


class WhisperSettingsTests(unittest.TestCase):
    def test_multilingual_model_transcribes_english(self):
        kwargs = whisper_generate_kwargs(fake_pipeline(True))
        self.assertEqual(kwargs["language"], "english")
        self.assertEqual(kwargs["task"], "transcribe")

    def test_english_only_model_gets_no_language_setting(self):
        kwargs = whisper_generate_kwargs(fake_pipeline(False))
        self.assertNotIn("language", kwargs)
        self.assertNotIn("task", kwargs)

    def test_aws_names_are_the_prompt(self):
        voice_pipeline = fake_pipeline(True)
        self.assertIn("prompt_ids", whisper_generate_kwargs(voice_pipeline))
        self.assertEqual(voice_pipeline.tokenizer.prompts, [AWS_VOCABULARY])
        self.assertNotIn("prompt_ids", whisper_generate_kwargs(voice_pipeline, vocabulary=None))


class TranscriptTextTests(unittest.TestCase):
    def test_spoken_answer_is_kept(self):
        self.assertEqual(
            transcript_text({"text": " The CloudTrail record shows the IAM principal. "}),
            "The CloudTrail record shows the IAM principal.",
        )

    def test_silence_filler_is_not_an_answer(self):
        for text in ["", "  ", " you", "You.", "Thank you.", "Thanks for watching!"]:
            with self.subTest(text=text), self.assertRaisesRegex(ValueError, "No speech was heard"):
                transcript_text({"text": text})


if __name__ == "__main__":
    unittest.main()
