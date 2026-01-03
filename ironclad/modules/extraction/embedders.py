# embedders.py
from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Optional, Tuple
import numpy as np
from PIL import Image
import torch

class BaseEmbedder(ABC):
    @abstractmethod
    def embed(self, img: Image.Image) -> Optional[np.ndarray]:
        """Return (512,) float32 L2-normalized embedding or None if no face."""
        ...

# -------- VGGFace2 (facenet_pytorch) --------
class VGGFace2Embedder(BaseEmbedder):
    def __init__(self):
        import torch
        from facenet_pytorch import InceptionResnetV1, MTCNN, fixed_image_standardization
        from torchvision import transforms

        # Pick Apple GPU if available
        # self.device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
        self.device = torch.device("cpu")
        # Fast, single-face detector/aligner (tune thresholds/min_face_size for speed)
        self.mtcnn = MTCNN(
            image_size=160, margin=10, post_process=True,
            keep_all=False,            # only largest face
            min_face_size=40,          # skip tiny faces
            thresholds=(0.9, 0.9, 0.9),
            device=self.device
        )

        self.model = InceptionResnetV1(pretrained="vggface2").eval().to(self.device)

        # For chips: replicate the same preprocessing MTCNN does
        self._chip_tf = transforms.Compose([
            transforms.ToTensor(),            # HWC [0,255] -> CHW [0,1]
            fixed_image_standardization       # per-image standardization
        ])

    def _forward_tensor(self, x):
        import numpy as np, torch
        with torch.no_grad():
            x = x.to(self.device)
            emb = self.model(x)[0].detach().cpu().numpy()
            emb = emb / (np.linalg.norm(emb) + 1e-12)
            return emb.astype("float32")

    def embed(self, img, assume_aligned: bool = False):
        """
        If assume_aligned=True, 'img' should be a pre-aligned face chip ~160x160.
        We skip MTCNN and feed the backbone directly (faster).
        """
        from PIL import Image
        import numpy as np
        import torch

        if assume_aligned:
            # Accept PIL or numpy
            if not isinstance(img, Image.Image):
                img = Image.fromarray(img)
            # Ensure 160x160 for InceptionResnetV1
            if img.size != (160, 160):
                img = img.resize((160, 160), Image.BILINEAR)
            x = self._chip_tf(img).unsqueeze(0)   # (1,3,160,160)
            return self._forward_tensor(x)

        # Default (full) path: detect + align + standardize via MTCNN
        with torch.no_grad():
            x = self.mtcnn(img)                   # (3,160,160) tensor or None
            if x is None:
                return None
            if x.ndim == 3:
                import torch as _torch
                x = x.unsqueeze(0)
            return self._forward_tensor(x)

# -------- ArcFace (InsightFace) --------
class ArcFaceEmbedder(BaseEmbedder):
    """
    Uses InsightFace FaceAnalysis (ONNX) to detect+align+embed.
    """
    # def __init__(self, ctx_id: int = -1, det_size: Tuple[int,int]=(640,640), model_pack: str = "buffalo_l"):
    #     from insightface.app import FaceAnalysis
    #     self.app = FaceAnalysis(name=model_pack)
    #     self.app.prepare(ctx_id=ctx_id, det_size=det_size)

    def __init__(self,
                 ctx_id: int = -1,
                 det_size: tuple[int,int] = (256, 256),  # smaller = much faster
                 model_pack: str = "buffalo_l",
                 prefer_coreml: bool = True):
        from insightface.app import FaceAnalysis
        self.app = FaceAnalysis(name=model_pack)
        providers = ['CoreMLExecutionProvider', 'CPUExecutionProvider'] if prefer_coreml else None
        try:
            self.app.prepare(
                ctx_id=ctx_id,
                det_size=det_size,
                providers=providers,
                allowed_modules=['detection', 'recognition']  # skip gender/age
            )
        except TypeError:
            # older insightface: no providers/allowed_modules args
            self.app.prepare(ctx_id=ctx_id, det_size=det_size)

    def embed(self, img: Image.Image) -> Optional[np.ndarray]:
        import cv2
        rgb = np.array(img.convert("RGB"))
        bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        faces = self.app.get(bgr)
        if not faces:
            return None
        # choose largest
        areas = [int((f.bbox[2]-f.bbox[0])*(f.bbox[3]-f.bbox[1])) for f in faces]
        f = faces[int(np.argmax(areas))]
        return f.normed_embedding.astype("float32")  # already L2-normalized