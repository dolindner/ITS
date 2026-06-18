import torch
from enum import Enum

from src.utils.transforms.rotation import (
    Rotation_2D,
    RotationComplex2D, RotationX3D, RotationY3D,
    RotationZ3D, RotationQuaternion3D, Rotation2Vec3D, Rotation3DEulerUniform,
)
from src.utils.transforms.scale import (
    Scale2D, Scale3D, UniformScale2D, UniformScale3D, ScaleX2D, ScaleY2D, ScaleZ3D, ScaleX3D, ScaleY3D, Reflection
)
from src.utils.transforms.shear import (
    Shear2DGeneral, Shear3DGeneral, ShearX2D, ShearY2D, ShearXY3D, ShearXZ3D, ShearYZ3D, ShearYX3D, ShearZX3D,
    ShearZY3D, ShearSequential2D, ShearSequential3D, ShearScaled
)
from src.utils.transforms.translation import (
    Translate2D, Translate3D, TranslateX2D, TranslateY2D, TranslateX3D, TranslateY3D, TranslateZ3D
)


class AffineTransformation2D(Enum):
    """
    Enum for 2D affine transformations.
    
    Example usage:
        transform = AffineTransformation2D.TRANSLATION.value
        matrix = transform.matrix(params)
        
    For backward compatibility:
        transform_dict = AffineTransformation2D.TRANSLATION.value.as_dict()
        matrix = transform_dict['matrix'](params)
    """
    TRANSLATION = Translate2D
    TRANSLATION_X = TranslateX2D
    TRANSLATION_Y = TranslateY2D

    ROTATION = Rotation_2D
    ROTATION_COMPLEX = RotationComplex2D

    SCALING = Scale2D
    SCALING_UNIFORM = UniformScale2D
    SCALING_X = ScaleX2D
    SCALING_Y = ScaleY2D

    SHEARING = Shear2DGeneral
    SHEARING_X = ShearX2D
    SHEARING_Y = ShearY2D
    SHEARSCALE = ShearScaled(dims=2)

    SHEARING_SEQUENTIAL = ShearSequential2D

    REFLECTION_X = Reflection(dims=2, axis=0)
    REFLECTION_Y = Reflection(dims=2, axis=1)


class AffineTransformations3D(Enum):
    """
    Enum for 3D affine transformations.
    
    Example usage:
        transform = AffineTransformations3D.TRANSLATION.value
        matrix = transform.matrix(params)
        
    For backward compatibility:
        transform_dict = AffineTransformations3D.TRANSLATION.value.as_dict()
        matrix = transform_dict['matrix'](params)
    """
    TRANSLATION = Translate3D
    TRANSLATION_X = TranslateX3D
    TRANSLATION_Y = TranslateY3D
    TRANSLATION_Z = TranslateZ3D

    ROTATIONEULER = Rotation3DEulerUniform()
    ROTATION = Rotation3DEulerUniform()

    ROTATION_X = RotationX3D
    ROTATION_Y = RotationY3D
    ROTATION_Z = RotationZ3D

    ROTATION2VEC3D = Rotation2Vec3D

    ROTATION_QUATERNION = RotationQuaternion3D

    SCALING = Scale3D
    SCALING_UNIFORM = UniformScale3D
    SCALING_X = ScaleX3D
    SCALING_Y = ScaleY3D
    SCALING_Z = ScaleZ3D

    SHEARING = Shear3DGeneral

    SHEARING_XY = ShearXY3D
    SHEARING_XZ = ShearXZ3D
    SHEARING_YZ = ShearYZ3D
    SHEARING_YX = ShearYX3D
    SHEARING_ZX = ShearZX3D
    SHEARING_ZY = ShearZY3D

    SHEARING_SEQUENTIAL = ShearSequential3D

    REFLECTION_X = Reflection(dims=3, axis=0)
    REFLECTION_Y = Reflection(dims=3, axis=1)
    REFLECTION_Z = Reflection(dims=3, axis=2)


if __name__ == '__main__':
    import torch

    # Test class-based transformations
    print("Testing class-based transformations")

    # 2D transformations
    params_2d_rotation = torch.tensor([[0.5]])
    rotation_matrix_2d = AffineTransformation2D.ROTATION.value["matrix"](None, params_2d_rotation)
    print("Rotation Matrix 2D:")
    print(rotation_matrix_2d)
