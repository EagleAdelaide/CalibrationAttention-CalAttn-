import torch
import timm
from torch import nn

from models.calattn import CalAttnWrapper
from models.relaxed_softmax import RelaxedSoftmaxWrapper


def build_backbone(cfg):
    name = cfg["model"]["backbone"]
    pretrained = bool(cfg["model"].get("pretrained", True))
    num_classes = int(cfg["model"]["num_classes"])

    # timm model with classifier head
    model = timm.create_model(name, pretrained=pretrained, num_classes=num_classes)
    return model


def build_model_and_wrapper(cfg):
    method = cfg["method"]["name"].lower()
    backbone = build_backbone(cfg)

    if method == "calattn":
        return CalAttnWrapper(
            backbone=backbone,
            hidden=int(cfg["method"].get("calattn_hidden", 128)),
            mode=str(cfg["method"].get("calattn_on", "cls"))
        )
    if method == "relaxed_softmax":
        return RelaxedSoftmaxWrapper(
            backbone=backbone,
            hidden=int(cfg["method"].get("relaxed_hidden", 128)),
        )

    # default: return backbone (CE/BS/MMCE/DFL etc. operate on logits)
    return backbone
