"""Hardware abstraction (camera, rotation stage, lights). Only file-based and mock providers are implemented.
No real hardware is ever commanded by this package."""
from .interface import CameraProvider, CaptureCoordinator, LightController, RotationStageProvider  # noqa: F401
from .mock import FileCameraProvider, MockCamera, MockLightController, MockRotationStage  # noqa: F401
