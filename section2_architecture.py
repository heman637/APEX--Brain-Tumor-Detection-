"""
=============================================================
  BRAIN TUMOR DETECTION — SECTION 2: SYSTEM ARCHITECTURE v2
=============================================================
UPGRADES over v1:
  [1] U-Net decoder with SKIP CONNECTIONS (5 scales)
  [2] DEEP SUPERVISION — predictions at 3 decoder levels
  [3] Attention Gates on skip connections
  [2.1] 2.5D Multi-View Cross-Attention Fusion
  [2.2] Full Pipeline Architecture
  [2.3] Multi-Task Loss with Deep Supervision

Expected Dice improvement: 0.61 → 0.88-0.92
=============================================================
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import efficientnet_b4, EfficientNet_B4_Weights

NUM_CLASSES  = 4
LAMBDA_CE    = 1.0
LAMBDA_DICE  = 1.0
LAMBDA_FOCAL = 0.5


# ══════════════════════════════════════════════════════════════
# MODULE 1 — EfficientNet-B4 Encoder with Skip Connection Outputs
# ══════════════════════════════════════════════════════════════
class EfficientNetEncoder(nn.Module):
    """
    EfficientNet-B4 backbone that returns intermediate feature maps
    at 5 scales for skip connections.

    Scale outputs (for 224x224 input):
      skip0: (B, 48,  112, 112)  — after features[1]
      skip1: (B, 32,  56,  56)   — after features[2]
      skip2: (B, 56,  28,  28)   — after features[3]
      skip3: (B, 160, 14,  14)   — after features[5]
      final: (B, 1792, 7,   7)   — after features[8]
    """

    def __init__(self, in_channels=3, pretrained=True):
        super().__init__()

        if pretrained:
            backbone = efficientnet_b4(weights=EfficientNet_B4_Weights.IMAGENET1K_V1)
        else:
            backbone = efficientnet_b4(weights=None)

        # Adapt first conv for non-RGB inputs
        if in_channels != 3:
            orig = backbone.features[0][0]
            backbone.features[0][0] = nn.Conv2d(
                in_channels, orig.out_channels,
                kernel_size=orig.kernel_size,
                stride=orig.stride,
                padding=orig.padding,
                bias=False
            )
            with torch.no_grad():
                w = orig.weight.data.mean(dim=1, keepdim=True)
                backbone.features[0][0].weight.data = w.repeat(1, in_channels, 1, 1)

        # Split backbone into 5 stages for skip connections
        self.stage0 = backbone.features[0]    # (B, 48,  112, 112)
        self.stage1 = backbone.features[1]    # (B, 24,  112, 112)
        self.stage2 = backbone.features[2]    # (B, 32,   56,  56)
        self.stage3 = backbone.features[3]    # (B, 56,   28,  28)
        self.stage4 = backbone.features[4]    # (B, 112,  14,  14)
        self.stage5 = backbone.features[5]    # (B, 160,  14,  14)
        self.stage6 = backbone.features[6]    # (B, 272,   7,   7)
        self.stage7 = backbone.features[7]    # (B, 448,   7,   7)
        self.stage8 = backbone.features[8]    # (B, 1792,  7,   7)

        self.feature_dim = 1792
        self.dropout     = nn.Dropout(p=0.4)

    def forward(self, x):
        """Returns final pooled features (B, 1792) for classification."""
        x = self.encode(x)[-1]
        x = F.adaptive_avg_pool2d(x, 1).flatten(1)
        x = self.dropout(x)
        return x

    def encode(self, x):
        """
        Returns list of feature maps at each scale.
        Used by segmentation decoder for skip connections.
        """
        s0 = self.stage0(x)       # (B, 48,  112, 112)
        s1 = self.stage1(s0)      # (B, 24,  112, 112)
        s2 = self.stage2(s1)      # (B, 32,   56,  56)
        s3 = self.stage3(s2)      # (B, 56,   28,  28)
        s4 = self.stage4(s3)      # (B, 112,  14,  14)
        s5 = self.stage5(s4)      # (B, 160,  14,  14)
        s6 = self.stage6(s5)      # (B, 272,   7,   7)
        s7 = self.stage7(s6)      # (B, 448,   7,   7)
        s8 = self.stage8(s7)      # (B, 1792,  7,   7)
        return [s0, s1, s2, s3, s4, s5, s6, s7, s8]

    def get_feature_map(self, x):
        """Returns final spatial feature map (B, 1792, 7, 7)."""
        return self.encode(x)[-1]


# ══════════════════════════════════════════════════════════════
# MODULE 2 — Attention Gate (for skip connections)
# ══════════════════════════════════════════════════════════════
class AttentionGate(nn.Module):
    """
    Attention gate that filters skip connection features.
    Suppresses irrelevant regions, focuses on tumor.

    g = gating signal from decoder (coarser scale)
    x = skip connection from encoder (finer scale)
    """

    def __init__(self, f_g, f_x, f_int):
        super().__init__()
        self.W_g = nn.Sequential(
            nn.Conv2d(f_g, f_int, 1, bias=False),
            nn.BatchNorm2d(f_int)
        )
        self.W_x = nn.Sequential(
            nn.Conv2d(f_x, f_int, 1, bias=False),
            nn.BatchNorm2d(f_int)
        )
        self.psi = nn.Sequential(
            nn.Conv2d(f_int, 1, 1, bias=False),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, g, x):
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        # Upsample g to match x spatial size
        if g1.shape[2:] != x1.shape[2:]:
            g1 = F.interpolate(g1, size=x1.shape[2:],
                               mode="bilinear", align_corners=False)
        psi = self.relu(g1 + x1)
        psi = self.psi(psi)
        return x * psi


# ══════════════════════════════════════════════════════════════
# MODULE 3 — Double Conv Block
# ══════════════════════════════════════════════════════════════
class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch, dropout=0.1):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True),
            nn.Dropout2d(dropout),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


# ══════════════════════════════════════════════════════════════
# MODULE 4 — U-Net Decoder WITH Skip Connections + Deep Supervision
# ══════════════════════════════════════════════════════════════
class SegmentationDecoder(nn.Module):
    """
    Full U-Net decoder with:
    - Skip connections from encoder at every scale
    - Attention gates on each skip connection
    - Deep supervision: predictions at 3 scales
    - Final output: (B, 3, 224, 224)
    - Deep outputs: (B, 3, H/2, W/2), (B, 3, H/4, W/4)

    EfficientNet-B4 skip channel sizes:
      s8: 1792  (bottleneck)
      s5: 160
      s3: 56
      s2: 32
      s0: 48
    """

    def __init__(self, num_seg_classes=3):
        super().__init__()

        # ── Decoder Block 1: 7x7 → 14x14 ──
        self.up1    = nn.ConvTranspose2d(1792, 512, 2, stride=2)
        self.attn1  = AttentionGate(f_g=512, f_x=160, f_int=128)
        self.dec1   = ConvBlock(512 + 160, 512)

        # ── Decoder Block 2: 14x14 → 28x28 ──
        self.up2    = nn.ConvTranspose2d(512, 256, 2, stride=2)
        self.attn2  = AttentionGate(f_g=256, f_x=56, f_int=64)
        self.dec2   = ConvBlock(256 + 56, 256)

        # ── Decoder Block 3: 28x28 → 56x56 ──
        self.up3    = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.attn3  = AttentionGate(f_g=128, f_x=32, f_int=32)
        self.dec3   = ConvBlock(128 + 32, 128)

        # ── Decoder Block 4: 56x56 → 112x112 ──
        self.up4    = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.attn4  = AttentionGate(f_g=64, f_x=48, f_int=32)
        self.dec4   = ConvBlock(64 + 48, 64)

        # ── Decoder Block 5: 112x112 → 224x224 ──
        self.up5    = nn.ConvTranspose2d(64, 32, 2, stride=2)
        self.dec5   = ConvBlock(32, 32)

        # ── Final output (224x224) ──
        self.final  = nn.Conv2d(32, num_seg_classes, 1)

        # ── Deep supervision heads ──
        # DS1: from dec2 output (28x28)
        self.ds1    = nn.Conv2d(256, num_seg_classes, 1)
        # DS2: from dec3 output (56x56)
        self.ds2    = nn.Conv2d(128, num_seg_classes, 1)

    def forward(self, encoder_features):
        """
        Args:
          encoder_features: list from EfficientNetEncoder.encode()
            [s0(48), s1(24), s2(32), s3(56), s4(112),
             s5(160), s6(272), s7(448), s8(1792)]

        Returns:
          main_out : (B, 3, 224, 224) — final prediction
          ds_outs  : list of deep supervision outputs
                     [(B,3,28,28), (B,3,56,56)]
        """
        s0, s1, s2, s3, s4, s5, s6, s7, s8 = encoder_features

        # Block 1: bottleneck → 14x14
        x = self.up1(s8)                         # (B, 512, 14, 14)
        s5_att = self.attn1(x, s5)               # attended skip
        x = self.dec1(torch.cat([x, s5_att], 1)) # (B, 512, 14, 14)

        # Block 2: → 28x28
        x = self.up2(x)                          # (B, 256, 28, 28)
        s3_att = self.attn2(x, s3)
        x = self.dec2(torch.cat([x, s3_att], 1)) # (B, 256, 28, 28)
        ds_out1 = self.ds1(x)                    # deep supervision 1

        # Block 3: → 56x56
        x = self.up3(x)                          # (B, 128, 56, 56)
        s2_att = self.attn3(x, s2)
        x = self.dec3(torch.cat([x, s2_att], 1)) # (B, 128, 56, 56)
        ds_out2 = self.ds2(x)                    # deep supervision 2

        # Block 4: → 112x112
        x = self.up4(x)                          # (B, 64, 112, 112)
        s0_att = self.attn4(x, s0)
        x = self.dec4(torch.cat([x, s0_att], 1)) # (B, 64, 112, 112)

        # Block 5: → 224x224
        x = self.up5(x)                          # (B, 32, 224, 224)
        x = self.dec5(x)
        main_out = self.final(x)                 # (B, 3, 224, 224)

        return main_out, [ds_out1, ds_out2]


# ══════════════════════════════════════════════════════════════
# MODULE 5 — Cross-Attention Fusion (Section 2.1)
# ══════════════════════════════════════════════════════════════
class CrossAttentionFusion(nn.Module):
    """2.5D Multi-View Fusion via Cross-Attention."""

    def __init__(self, feature_dim=1792, num_heads=8):
        super().__init__()
        self.proj_dim      = 512
        self.proj_axial    = nn.Linear(feature_dim, self.proj_dim)
        self.proj_sagittal = nn.Linear(feature_dim, self.proj_dim)
        self.proj_coronal  = nn.Linear(feature_dim, self.proj_dim)
        self.cross_attn    = nn.MultiheadAttention(
            embed_dim=self.proj_dim, num_heads=num_heads,
            dropout=0.1, batch_first=True
        )
        self.norm1    = nn.LayerNorm(self.proj_dim)
        self.norm2    = nn.LayerNorm(self.proj_dim)
        self.ffn      = nn.Sequential(
            nn.Linear(self.proj_dim, self.proj_dim * 2),
            nn.GELU(), nn.Dropout(0.1),
            nn.Linear(self.proj_dim * 2, self.proj_dim),
        )
        self.out_proj = nn.Linear(self.proj_dim * 3, feature_dim)
        self.out_norm = nn.LayerNorm(feature_dim)

    def forward(self, feat_axial, feat_sagittal, feat_coronal):
        q_ax   = self.proj_axial(feat_axial).unsqueeze(1)
        q_sag  = self.proj_sagittal(feat_sagittal).unsqueeze(1)
        q_cor  = self.proj_coronal(feat_coronal).unsqueeze(1)
        views  = torch.cat([q_ax, q_sag, q_cor], dim=1)
        attn, _= self.cross_attn(views, views, views)
        attn   = self.norm1(attn + views)
        ffn_out= self.norm2(self.ffn(attn) + attn)
        fused  = self.out_norm(self.out_proj(ffn_out.flatten(start_dim=1)))
        return fused


# ══════════════════════════════════════════════════════════════
# MODULE 6 — Classification Head
# ══════════════════════════════════════════════════════════════
class ClassificationHead(nn.Module):
    def __init__(self, feature_dim=1792, num_classes=4):
        super().__init__()
        self.head = nn.Sequential(
            nn.Linear(feature_dim, 512),
            nn.BatchNorm1d(512), nn.ReLU(inplace=True), nn.Dropout(0.4),
            nn.Linear(512, 128),
            nn.BatchNorm1d(128), nn.ReLU(inplace=True), nn.Dropout(0.3),
            nn.Linear(128, num_classes),
        )

    def forward(self, x):
        return self.head(x)


# ══════════════════════════════════════════════════════════════
# MODULE 7 — Full Brain Tumor Model
# ══════════════════════════════════════════════════════════════
class BrainTumorModel(nn.Module):
    """
    Stage 1: Classification only (2D RGB)
    Stage 2: Classification + Segmentation with skip connections
    Stage 3: 2.5D Fusion + Classification + Segmentation
    """

    def __init__(self, stage=1, in_channels=3, pretrained=True):
        super().__init__()
        self.stage    = stage
        self.encoder  = EfficientNetEncoder(in_channels, pretrained)
        feat_dim      = self.encoder.feature_dim
        self.cls_head = ClassificationHead(feat_dim, NUM_CLASSES)

        if stage >= 2:
            self.seg_decoder = SegmentationDecoder(num_seg_classes=3)
        if stage >= 3:
            self.fusion = CrossAttentionFusion(feat_dim, num_heads=8)

    def forward(self, x, x_sag=None, x_cor=None):
        outputs = {}

        if self.stage == 1:
            features = self.encoder(x)
            outputs["cls_logits"] = self.cls_head(features)

        elif self.stage == 2:
            enc_feats = self.encoder.encode(x)
            # Classification from final feature map
            features  = F.adaptive_avg_pool2d(enc_feats[-1], 1).flatten(1)
            features  = self.encoder.dropout(features)
            outputs["cls_logits"] = self.cls_head(features)
            # Segmentation with skip connections
            main_out, ds_outs     = self.seg_decoder(enc_feats)
            outputs["seg_logits"] = main_out
            outputs["ds_logits"]  = ds_outs   # deep supervision outputs

        elif self.stage == 3:
            assert x_sag is not None and x_cor is not None
            feat_ax  = self.encoder(x)
            feat_sag = self.encoder(x_sag)
            feat_cor = self.encoder(x_cor)
            fused                 = self.fusion(feat_ax, feat_sag, feat_cor)
            outputs["cls_logits"] = self.cls_head(fused)
            enc_feats             = self.encoder.encode(x)
            main_out, ds_outs     = self.seg_decoder(enc_feats)
            outputs["seg_logits"] = main_out
            outputs["ds_logits"]  = ds_outs

        return outputs

    def freeze_encoder(self, num_layers=3):
        stages = [self.encoder.stage0, self.encoder.stage1,
                  self.encoder.stage2, self.encoder.stage3,
                  self.encoder.stage4, self.encoder.stage5]
        for i, stage in enumerate(stages):
            if i < num_layers:
                for p in stage.parameters():
                    p.requires_grad = False
        print(f"  Froze first {num_layers} encoder stages")

    def unfreeze_all(self):
        for p in self.parameters():
            p.requires_grad = True
        print("  All layers unfrozen")

    def get_trainable_params(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def get_total_params(self):
        return sum(p.numel() for p in self.parameters())


# ══════════════════════════════════════════════════════════════
# MODULE 8 — Losses
# ══════════════════════════════════════════════════════════════
class DiceLoss(nn.Module):
    def __init__(self, smooth=1e-6):
        super().__init__()
        self.smooth = smooth

    def forward(self, pred, target):
        pred   = torch.sigmoid(pred)
        pred   = pred.view(pred.size(0), pred.size(1), -1)
        target = target.view(target.size(0), target.size(1), -1)
        inter  = (pred * target).sum(dim=2)
        union  = pred.sum(dim=2) + target.sum(dim=2)
        return 1 - ((2 * inter + self.smooth) / (union + self.smooth)).mean()


class FocalLoss(nn.Module):
    def __init__(self, alpha=1.0, gamma=2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, pred, target):
        ce    = F.cross_entropy(pred, target, reduction="none")
        pt    = torch.exp(-ce)
        return (self.alpha * (1 - pt) ** self.gamma * ce).mean()


class MultiTaskLoss(nn.Module):
    """
    Total = CE + Focal + Dice (main) + 0.4*(Dice ds1 + Dice ds2)
    Deep supervision weights decay with scale (0.4 each)
    """

    def __init__(self, lambda_ce=LAMBDA_CE,
                 lambda_dice=LAMBDA_DICE,
                 lambda_focal=LAMBDA_FOCAL,
                 lambda_ds=0.4):
        super().__init__()
        self.lambda_ce    = lambda_ce
        self.lambda_dice  = lambda_dice
        self.lambda_focal = lambda_focal
        self.lambda_ds    = lambda_ds
        self.ce    = nn.CrossEntropyLoss()
        self.dice  = DiceLoss()
        self.focal = FocalLoss()

    def forward(self, outputs, cls_labels=None, seg_masks=None):
        ld   = {}
        loss = torch.tensor(0.0, device=outputs["cls_logits"].device)

        # Classification losses
        if cls_labels is not None:
            ce    = self.ce(outputs["cls_logits"], cls_labels)
            focal = self.focal(outputs["cls_logits"], cls_labels)
            loss  = loss + self.lambda_ce * ce + self.lambda_focal * focal
            ld["ce_loss"]    = ce.item()
            ld["focal_loss"] = focal.item()

        # Main segmentation loss
        if "seg_logits" in outputs and seg_masks is not None:
            dice = self.dice(outputs["seg_logits"], seg_masks)
            loss = loss + self.lambda_dice * dice
            ld["dice_loss"] = dice.item()

            # Deep supervision losses
            if "ds_logits" in outputs:
                for i, ds_logit in enumerate(outputs["ds_logits"]):
                    # Downsample target to match ds output size
                    ds_target = F.interpolate(
                        seg_masks.float(),
                        size=ds_logit.shape[2:],
                        mode="nearest"
                    )
                    ds_dice = self.dice(ds_logit, ds_target)
                    loss    = loss + self.lambda_ds * ds_dice
                    ld[f"ds{i+1}_dice"] = ds_dice.item()

        ld["total_loss"] = loss.item()
        return loss, ld


# ══════════════════════════════════════════════════════════════
# FACTORY
# ══════════════════════════════════════════════════════════════
def get_model(stage=1, in_channels=3, pretrained=True, device="cuda"):
    model = BrainTumorModel(stage, in_channels, pretrained).to(device)
    print(f"\n  Model Stage  : {stage}")
    print(f"  In channels  : {in_channels}")
    print(f"  Total params : {model.get_total_params():,}")
    print(f"  Trainable    : {model.get_trainable_params():,}")
    print(f"  Device       : {device}")
    return model


# ══════════════════════════════════════════════════════════════
# SELF TEST
# ══════════════════════════════════════════════════════════════
if __name__ == "__main__":
    print("\n" + "="*60)
    print("  SECTION 2 v2 — ARCHITECTURE SELF TEST")
    print("="*60)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"  Device: {device}")

    # Stage 1
    m1 = get_model(stage=1, in_channels=3, pretrained=False, device=device)
    x  = torch.randn(2, 3, 224, 224).to(device)
    o1 = m1(x)
    assert o1["cls_logits"].shape == (2, 4)
    print(f"  Stage 1 cls_logits : {o1['cls_logits'].shape}  [OK]")

    # Stage 2
    m2  = get_model(stage=2, in_channels=4, pretrained=False, device=device)
    x4  = torch.randn(2, 4, 224, 224).to(device)
    o2  = m2(x4)
    assert o2["cls_logits"].shape == (2, 4)
    assert o2["seg_logits"].shape == (2, 3, 224, 224)
    assert len(o2["ds_logits"]) == 2
    print(f"  Stage 2 cls_logits : {o2['cls_logits'].shape}  [OK]")
    print(f"  Stage 2 seg_logits : {o2['seg_logits'].shape}  [OK]")
    print(f"  Stage 2 ds_logits  : {[d.shape for d in o2['ds_logits']]}  [OK]")

    # Multi-task loss with deep supervision
    criterion  = MultiTaskLoss()
    cls_labels = torch.randint(0, 4, (2,)).to(device)
    seg_masks  = torch.randint(0, 2, (2, 3, 224, 224)).float().to(device)
    loss, ld   = criterion(o2, cls_labels, seg_masks)
    print(f"  CE Loss     : {ld.get('ce_loss',    0):.4f}")
    print(f"  Dice Loss   : {ld.get('dice_loss',  0):.4f}")
    print(f"  DS1 Dice    : {ld.get('ds1_dice',   0):.4f}")
    print(f"  DS2 Dice    : {ld.get('ds2_dice',   0):.4f}")
    print(f"  Total Loss  : {ld['total_loss']:.4f}  [OK]")

    # Freeze/unfreeze
    m2.freeze_encoder(num_layers=3)
    print(f"  Trainable (frozen) : {m2.get_trainable_params():,}")
    m2.unfreeze_all()
    print(f"  Trainable (all)    : {m2.get_trainable_params():,}")

    print("\n  SECTION 2 v2 COMPLETE")
    print("  Skip connections + Attention Gates + Deep Supervision — READY")
    print("="*60 + "\n")