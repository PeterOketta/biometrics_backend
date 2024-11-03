import numpy as np
import onnxruntime as ort


class ECGModelInference:
    def __init__(self, model_path="model_2.onnx"):
        self.session = ort.InferenceSession(model_path, providers=['CPUExecutionProvider'])
        self.input_name = self.session.get_inputs()[0].name

    def generate_embedding(self, segment):
        """
        Generate embedding for a single segment.
        segment should be a numpy array of shape (200,)
        """
        # Ensure segment is the right length
        if len(segment) != 200:
            raise ValueError(f"Segment must be length 200, got {len(segment)}")
            
        # Reshape to [1, 200, 1] - adding batch and channel dimensions
        segment_reshaped = np.array(segment, dtype=np.float32).reshape(1, 200, 1)
        
        # Run inference
        embedding = self.session.run(None, {self.input_name: segment_reshaped})[0]
        return embedding.flatten()