from src.utils.affine_transforms import AffineTransformations3D,AffineTransformation2D
from src.utils.transform_sequence import TransformSequence
from src.utils.transformation_problem import TransformationProblem
from src.utils.transforms.scale import Reflection

def replace_rotation_transforms(problem):
    """
    Creates a copy of the problem with rotation transformations replaced.
    - 2D rotations are replaced by ROTATION_COMPLEX
    - 3D rotations are replaced by ROTATION_QuaTERNION
    - Domains for rotation transforms are set to None

    Args:
        problem: TransformationProblem or TransformSequence to modify

    Returns:
        New TransformationProblem or TransformSequence with rotations replaced
    """
    # Accept either a TransformationProblem or a TransformSequence
    is_sequence_input = isinstance(problem, TransformSequence)
    if is_sequence_input:
        original_sequence = problem
    else:
        original_sequence = problem.transform_sequence

    transformations = original_sequence.transformations
    domains = original_sequence.domains

    # Check if modification is needed
    needs_update = False
    for t in transformations:
        # Check for 2D rotation (enum value or class)
        if  t == AffineTransformation2D.ROTATION.value:
            needs_update = True
            break
        # Check for 3D rotation (enum value or class)
        if  t == AffineTransformations3D.ROTATION.value:
            needs_update = True
            break
        if  t == AffineTransformations3D.ROTATIONEULER.value:
            needs_update = True
            break



    if not needs_update:
        # assert that the transfroms are not instances of the classes
        for t in transformations:
            if isinstance(t, type(AffineTransformation2D.ROTATION)) or isinstance(t, type(AffineTransformations3D.ROTATION)) or isinstance(t, type(AffineTransformations3D.ROTATION_EULER)):
                assert False, "Transformations contain instances of rotation classes, but where not replaced."
        return original_sequence if is_sequence_input else problem

    # Build new transformations and domains
    new_transformations = []
    new_domains = []

    for i, t in enumerate(transformations):
        replaced = False

        # Check for 2D rotation
        if  t == AffineTransformation2D.ROTATION.value:
            new_transformations.append(AffineTransformation2D.ROTATION_COMPLEX.value)
            new_domains.append(None)
            replaced = True

        # Check for 3D rotation (standard)
        if not replaced and t == AffineTransformations3D.ROTATION.value:
            new_transformations.append(AffineTransformations3D.ROTATION_QUATERNION.value)
            new_domains.append(None)
            replaced = True

        # Check for 3D Euler rotation
        if not replaced and t == AffineTransformations3D.ROTATIONEULER.value:
            new_transformations.append(AffineTransformations3D.ROTATION_QUATERNION.value)
            new_domains.append(None)
            replaced = True

        # Keep original if not replaced
        if not replaced:
            new_transformations.append(t)
            new_domains.append(domains[i])

    # Create new TransformSequence
    new_sequence = TransformSequence(
        transformations=new_transformations,
        domains=new_domains,
        neighbour_hood_size=original_sequence.neighbour_hood_size,
        application_method=original_sequence.application_method,
        device=original_sequence.dummy_param.device,
        dtype=original_sequence.dummy_param.dtype,
        init_method=original_sequence.init_method,
        reflect=original_sequence.reflect,
        invert=original_sequence.invert
    )

    for t in new_transformations:
        if isinstance(t, type(AffineTransformation2D.ROTATION) or isinstance(t,
                                                                             type(AffineTransformations3D.ROTATION)) or isinstance(
                t, type(AffineTransformations3D.ROTATIONEULER))):
            assert False, "Transformations contain instances of rotation classes, but where not replaced."

    # Return same type as input
    if is_sequence_input:
        return new_sequence

    # Create new TransformationProblem (original behavior)
    new_problem = TransformationProblem(
        confidence_module=problem.confidence_module,
        transform_sequence=new_sequence,
        consolidate_method=problem.consolidate_method,
        max_batch_size=problem.max_batch_size
    )

    return new_problem

def replace_rotation_transforms_2vec(problem):
    """
    Creates a copy of the problem with rotation transformations replaced.
    - 2D rotations are replaced by ROTATION_COMPLEX
    - 3D rotations are replaced by ROTATION 2VEC3D
    - Domains for rotation transforms are set to None

    Args:
        problem: TransformationProblem or TransformSequence to modify

    Returns:
        New TransformationProblem or TransformSequence with rotations replaced
    """
    # Accept either a TransformationProblem or a TransformSequence
    is_sequence_input = isinstance(problem, TransformSequence)
    if is_sequence_input:
        original_sequence = problem
    else:
        original_sequence = problem.transform_sequence

    transformations = original_sequence.transformations
    domains = original_sequence.domains

    # Check if modification is needed
    needs_update = False
    for t in transformations:
        # Check for 2D rotation (enum value or class)
        if t == AffineTransformation2D.ROTATION.value:
            needs_update = True
            break
        # Check for 3D rotation (enum value or class)
        if t == AffineTransformations3D.ROTATION.value:
            needs_update = True
            break
        if t == AffineTransformations3D.ROTATIONEULER.value:
            needs_update = True
            break

    if not needs_update:
        return original_sequence if is_sequence_input else problem

    # Build new transformations and domains
    new_transformations = []
    new_domains = []

    for i, t in enumerate(transformations):
        replaced = False

        # Check for 2D rotation
        if t == AffineTransformation2D.ROTATION.value:
            new_transformations.append(AffineTransformation2D.ROTATION_COMPLEX.value)
            new_domains.append(None)
            replaced = True

        # Check for 3D rotation (standard)
        if not replaced and  t == AffineTransformations3D.ROTATION.value:
            new_transformations.append(AffineTransformations3D.ROTATION2VEC3D.value)
            new_domains.append(None)
            replaced = True

        # Check for 3D Euler rotation
        if not replaced and t == AffineTransformations3D.ROTATIONEULER.value:
            new_transformations.append(AffineTransformations3D.ROTATION2VEC3D.value)
            new_domains.append(None)
            replaced = True

        # Keep original if not replaced
        if not replaced:
            new_transformations.append(t)
            new_domains.append(domains[i])

    # Create new TransformSequence
    new_sequence = TransformSequence(
        transformations=new_transformations,
        domains=new_domains,
        neighbour_hood_size=original_sequence.neighbour_hood_size,
        application_method=original_sequence.application_method,
        device=original_sequence.dummy_param.device,
        dtype=original_sequence.dummy_param.dtype,
        init_method=original_sequence.init_method,
        reflect=original_sequence.reflect,
        invert=original_sequence.invert
    )
    for t in new_transformations:
        if isinstance(t, type(AffineTransformation2D.ROTATION) or isinstance(t,
                                                                             type(AffineTransformations3D.ROTATION)) or isinstance(
                t, type(AffineTransformations3D.ROTATIONEULER))):
            assert False, "Transformations contain instances of rotation classes, but where not replaced."


    # Return same type as input
    if is_sequence_input:
        return new_sequence

    # Create new TransformationProblem (original behavior)
    new_problem = TransformationProblem(
        confidence_module=problem.confidence_module,
        transform_sequence=new_sequence,
        consolidate_method=problem.consolidate_method,
        max_batch_size=problem.max_batch_size
    )

    return new_problem

from src.utils.transforms.scale import Scale,DirectedScale,ScaleAllSame

def replace_scaling_transforms(obj, replace_scale=False):
    """
    Replace 2D scaling transforms in either a TransformationProblem or TransformSequence.
    - Only handles 2D scaling variants (SCALING, SCALING_X, SCALING_Y, SCALING_UNIFORM).
    - When a scaling enum/value indicates log=True, replace it with the corresponding non-log Scale object.
      Otherwise the original transform is kept.

    Args:
        obj: TransformationProblem or TransformSequence
        replace_scale: whether to attempt scaling replacements

    Returns:
        Same type as input with scaling replacements applied (if any).
    """

    # accept either Problem or Sequence
    if isinstance(obj, TransformationProblem):
        sequence = obj.transform_sequence
    elif isinstance(obj, TransformSequence):
        sequence = obj
    else:
        raise TypeError("Input must be a TransformationProblem or TransformSequence")

    transformations = sequence.transformations
    domains = sequence.domains

    # quick check if any scaling present
    scaling_keys = {
        AffineTransformation2D.SCALING.value: "SCALING",
        AffineTransformation2D.SCALING_X.value: "SCALING_X",
        AffineTransformation2D.SCALING_Y.value: "SCALING_Y",
        AffineTransformation2D.SCALING_UNIFORM.value: "SCALING_UNIFORM",
        AffineTransformation2D.REFLECTION_X.value: "REFLECTION_X",
        AffineTransformation2D.REFLECTION_Y.value: "REFLECTION_Y",
    }
    needs_update = any(t in scaling_keys for t in transformations)
    if not needs_update:
        return obj

    new_transformations = []
    new_domains = []

    for t, d in zip(transformations, domains):
        if t == AffineTransformation2D.SCALING.value and replace_scale:
            # replace only if the enum/value indicated a log parameter previously
            if getattr(AffineTransformation2D.SCALING.value, "log", False):
                new_transformations.append(Scale(2, log=False))
                new_domains.append(d)
            else:
                new_transformations.append(t)
                new_domains.append(d)
        elif t == AffineTransformation2D.SCALING_X.value and replace_scale:
            if getattr(AffineTransformation2D.SCALING_X.value, "log", False):
                new_transformations.append(DirectedScale(2, 0, log=False))
                new_domains.append(d)
            else:
                new_transformations.append(t)
                new_domains.append(d)
        elif t == AffineTransformation2D.SCALING_Y.value and replace_scale:
            if getattr(AffineTransformation2D.SCALING_Y.value, "log", False):
                new_transformations.append(DirectedScale(2, 1, log=False))
                new_domains.append(d)
            else:
                new_transformations.append(t)
                new_domains.append(d)
        elif t == AffineTransformation2D.SCALING_UNIFORM.value and replace_scale:
            if getattr(AffineTransformation2D.SCALING_UNIFORM.value, "log", False):
                new_transformations.append(ScaleAllSame(2, log=False))
                new_domains.append(d)
            else:
                new_transformations.append(t)
                new_domains.append(d)
        elif t == AffineTransformation2D.REFLECTION_X.value:
            # reflections are a special case of scaling with log=False
            zw = Reflection(dims=2,axis=0)
            zw.eps=1
            new_transformations.append(zw)
            new_domains.append(d)
        elif t == AffineTransformation2D.REFLECTION_Y.value:
            zw = Reflection(dims=2,axis=1)
            zw.eps=1
            new_transformations.append(zw)
            new_domains.append(d)
        else:
            # leave everything else unchanged (including rotations)
            new_transformations.append(t)
            new_domains.append(d)

    # build new TransformSequence with same meta as original
    new_sequence = TransformSequence(
        transformations=new_transformations,
        domains=new_domains,
        neighbour_hood_size=sequence.neighbour_hood_size,
        application_method=sequence.application_method,
        device=sequence.dummy_param.device,
        dtype=sequence.dummy_param.dtype,
        init_method=sequence.init_method,
        reflect=sequence.reflect,
        invert=sequence.invert
    )

    # return same type as input
    if isinstance(obj, TransformationProblem):
        return TransformationProblem(
            confidence_module=obj.confidence_module,
            transform_sequence=new_sequence,
            consolidate_method=obj.consolidate_method,
            max_batch_size=obj.max_batch_size
        )
    else:
        return new_sequence