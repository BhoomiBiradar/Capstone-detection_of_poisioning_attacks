# Attack modules
from .label_flipping import apply_label_flipping_attack
from .corruption import apply_feature_corruption_attack
from .fgsm import apply_fgsm_attack

__all__ = [
    'apply_label_flipping_attack',
    'apply_feature_corruption_attack',
    'apply_fgsm_attack'
]

