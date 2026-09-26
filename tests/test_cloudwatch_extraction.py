"""Extraction contract checks; these do not measure live model accuracy."""
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from PIL import Image
from PIL.PngImagePlugin import PngInfo
from cloudir.ai_models import image_model


class CloudWatchExtractionTests(unittest.TestCase):
    def test_rows_keep_truncated_text_and_query_is_not_an_event_fact(self):
        result = image_model._parse_vlm_output(json.dumps({
            "event_rows": [{"timestamp": "2023-10-01T10:12:10Z",
                            "log_stream": "auth/identity-center",
                            "message": "ConsoleLogin for user from 185.220.10...",
                            "truncated": False}],
            "query_text": "Search unauthorized activity",
            "time_range": "2023-10-01 10:10-10:30 UTC",
            "matched_records": 5,
            "security_relevance": "strong",
        }), {"template": "cloudwatch"})
        self.assertEqual(len(result["event_rows"]), 1)  # Never fill to badge count.
        self.assertTrue(result["event_rows"][0]["truncated"])
        self.assertTrue(result["extraction_warnings"])
        self.assertEqual(result["time_range"], "2023-10-01 10:10-10:30 UTC")
        self.assertNotIn("unauthorized", " ".join(result["visible_facts_extracted"]))
        self.assertNotIn("security_relevance", result)

    def test_null_fields_remain_unknown_and_are_warned_about(self):
        result = image_model._normalise_cloudwatch_extraction({
            "event_rows": [{"timestamp": None, "log_stream": None,
                            "message": "visible message", "truncated": False}]})
        self.assertIsNone(result["event_rows"][0]["timestamp"])
        self.assertEqual(len(result["extraction_warnings"]), 2)

    def test_bad_rows_are_not_silently_coerced_to_facts(self):
        for rows in ("rows", ["header"], [{"timestamp": 42}],
                     [{"message": "text", "truncated": "false"}]):
            with self.subTest(rows=rows), self.assertRaises(ValueError):
                image_model._normalise_cloudwatch_extraction({"event_rows": rows})

    def test_wrapped_cell_lines_are_joined_into_one_message(self):
        result = image_model._normalise_cloudwatch_extraction({
            "event_rows": [{"timestamp": "2023-10-01T10:02:47Z", "log_stream": "auth/identity-center",
                            "message": "AssumeRole by\narn:aws:sts::123456789012:assumed-role/Org/s from\n10.0.4.19",
                            "truncated": False}],
            "query_text": "fields @message\n| sort @timestamp"})
        self.assertEqual(result["event_rows"][0]["message"],
                         "AssumeRole by arn:aws:sts::123456789012:assumed-role/Org/s from 10.0.4.19")
        self.assertEqual(result["query_text"], "fields @message | sort @timestamp")

    def test_legacy_crop_preserves_table_pixels_without_modifying_source(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "screenshot.png"
            source = Image.new("RGB", (1200, 760), "white")
            source.paste("black", (78, 406, 1130, 676))
            source.save(path)
            original = path.read_bytes()
            images, prompt = image_model._cloudwatch_images(path)
            # Legacy screenshots keep the full image, which holds the query editor.
            self.assertEqual(images[0]["image"], str(path))
            view = images[1]["image"]
            self.assertAlmostEqual(view.width / 1052, image_model._CLOUDWATCH_CARD_SCALE, delta=0.05)
            self.assertEqual(view.getpixel((100, 100)), (0, 0, 0))
            self.assertIn("Image 2 magnifies", prompt)
            self.assertEqual(path.read_bytes(), original)

    def test_unknown_layout_uses_whole_image(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "other.png"
            Image.new("RGB", (600, 400), "blue").save(path)
            images, _ = image_model._cloudwatch_images(path)
            self.assertEqual(len(images), 2)
            view = images[1]["image"]
            self.assertAlmostEqual(view.width / view.height, 1.5, delta=0.05)

    def test_card_crop_includes_query_editor_for_older_layout_boxes(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "card.png"
            source = Image.new("RGB", (1200, 928), "white")
            source.paste("red", (70, 226, 1130, 240))  # matched-records line
            source.paste("black", (70, 354, 1130, 874))
            metadata = PngInfo()
            # Saved before 24 Sep 2026: the box started at the metadata strip.
            metadata.add_text("cloudir_cloudwatch_detail_box", json.dumps([70, 354, 1130, 874]))
            source.save(path, pnginfo=metadata)
            images, prompt = image_model._cloudwatch_images(path)
            self.assertEqual(len(images), 1)
            view = images[0]["image"]
            scale = view.width / 1060
            self.assertAlmostEqual(view.height / scale, 874 - image_model._CLOUDWATCH_CARD_TOP, delta=3)
            self.assertEqual(view.getpixel((20, round(15 * scale))), (255, 0, 0))
            self.assertEqual((view.width % 28, view.height % 28), (0, 0))
            self.assertEqual((images[0]["resized_width"], images[0]["resized_height"]), view.size)
            self.assertIn("query editor", prompt)

    def test_block_diagonal_attention_matches_dense_masked_attention(self):
        import torch
        from transformers.models.qwen2_5_vl import modeling_qwen2_5_vl as qwen
        torch.manual_seed(0)
        attention = qwen.Qwen2_5_VLVisionSdpaAttention(64, num_heads=4).eval()
        dense = qwen.Qwen2_5_VLVisionSdpaAttention.forward
        for lengths in ([7, 13, 20], [16, 16, 16, 9, 16, 16, 4]):
            hidden = torch.randn(sum(lengths), 64)
            cu_seqlens = torch.tensor([0, *torch.tensor(lengths).cumsum(0).tolist()], dtype=torch.int32)
            angles = torch.randn(sum(lengths), 8)
            position = (torch.cat((angles, angles), -1).cos(), torch.cat((angles, angles), -1).sin())
            with torch.no_grad(), patch.object(image_model, "_VISION_QUERY_SLICE", 5):
                expected = dense(attention, hidden, cu_seqlens, position_embeddings=position)
                actual = image_model._block_diagonal_vision_attention(
                    attention, hidden, cu_seqlens, position_embeddings=position)
            self.assertLess((expected - actual).abs().max().item(), 1e-5)


class VisionMemoryTests(unittest.TestCase):
    def test_freed_gpu_memory_is_returned_during_generation(self):
        # Keeping freed key/value buffers until generation ended pushed Metal memory
        # to 32 GB on a 24 GB Mac while live tensors stayed under 10 GB.
        import torch

        processor = image_model._ReturnFreedMpsMemory(every=3)
        scores = torch.zeros(1, 4)
        with patch.object(torch.mps, "empty_cache") as empty_cache:
            outputs = [processor(None, scores) for _ in range(7)]

        self.assertEqual(empty_cache.call_count, 2)  # after tokens 3 and 6
        self.assertTrue(all(out is scores for out in outputs))  # never changes what the model writes

    def test_only_apple_gpus_get_the_release_step(self):
        self.assertEqual(image_model._memory_bounded_generation("cpu"), {})
        self.assertEqual(image_model._memory_bounded_generation("cuda"), {})
        self.assertIn("logits_processor", image_model._memory_bounded_generation("mps"))

if __name__ == "__main__":
    unittest.main()
