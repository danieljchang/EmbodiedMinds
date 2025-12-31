import torch
import torch.nn as nn


class ObjectEncoder(nn.Module):
    """
    Encodes 3D object representations into learned embeddings.
    
    Input: 3D object features (num_objects, 7)
    Output: Object embeddings (num_objects, embedding_dim)
    """
    
    def __init__(self, object_feature_dim: int = 7, embedding_dim: int = 256):
        """
        Args:
            object_feature_dim: Dimensionality of 3D features (should be 7)
            embedding_dim: Dimensionality of output embeddings
        """
        super().__init__()
        
        self.encoder = nn.Sequential(
            nn.Linear(object_feature_dim, 128),
            nn.LayerNorm(128),
            nn.ReLU(),
            nn.Linear(128, embedding_dim),
            nn.LayerNorm(embedding_dim),
        )
        
        self.embedding_dim = embedding_dim
    
    def forward(self, objects: torch.Tensor) -> torch.Tensor:
        """
        Encode objects to embeddings.
        
        Args:
            objects: (num_objects, 7) tensor or (B, num_objects, 7)
            
        Returns:
            (num_objects, embedding_dim) or (B, num_objects, embedding_dim)
        """
        original_shape = objects.shape
        
        # Handle batched input
        if len(original_shape) == 3:
            B, N, D = original_shape
            objects_flat = objects.reshape(B * N, D)
            embeddings_flat = self.encoder(objects_flat)
            return embeddings_flat.reshape(B, N, self.embedding_dim)
        else:
            return self.encoder(objects)

