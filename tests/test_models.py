import unittest
from src.encoders.vision_encoder import VisionEncoder
from src.encoders.text_encoder import TextEncoder
from src.fusion.fusion_module import FusionModule
from src.policy.policy_transformer import PolicyTransformer

class TestModelComponents(unittest.TestCase):

    def setUp(self):
        self.vision_encoder = VisionEncoder()
        self.text_encoder = TextEncoder()
        self.fusion_module = FusionModule()
        self.policy_transformer = PolicyTransformer()

    def test_vision_encoder_output_shape(self):
        input_data = ...  # Replace with appropriate test input
        output = self.vision_encoder(input_data)
        self.assertEqual(output.shape, (expected_shape))  # Replace with expected shape

    def test_text_encoder_output_shape(self):
        input_data = ...  # Replace with appropriate test input
        output = self.text_encoder(input_data)
        self.assertEqual(output.shape, (expected_shape))  # Replace with expected shape

    def test_fusion_module_output_shape(self):
        vision_output = ...  # Replace with appropriate test output from vision encoder
        text_output = ...  # Replace with appropriate test output from text encoder
        output = self.fusion_module(vision_output, text_output)
        self.assertEqual(output.shape, (expected_shape))  # Replace with expected shape

    def test_policy_transformer_output_shape(self):
        fusion_output = ...  # Replace with appropriate test output from fusion module
        output = self.policy_transformer(fusion_output)
        self.assertEqual(output.shape, (expected_shape))  # Replace with expected shape

if __name__ == '__main__':
    unittest.main()