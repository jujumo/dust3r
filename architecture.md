# CroCo v2 vs. DUSt3R — architecture & weight sharing

Two diagrams + a sharing table. DUSt3R (`AsymmetricCroCo3DStereo`,
`dust3r/model.py`) subclasses CroCo (`CroCoNet`, `croco/models/croco.py`) and
reuses its encoder/decoder weights, so it helps to see them side by side.

## 1. CroCo v2 (self-supervised pretraining)

One Siamese **encoder**, one **decoder**, one **pixel-reconstruction head**.
The asymmetry is *masking*: image 1 is ~90% masked and reconstructed using the
full image 2 as cross-attention "memory".

**Edge colours:** <span style="color:#2f6fdb">■ blue = img1 stream</span> · <span style="color:#d9822b">■ orange = img2 stream</span> · <span style="color:#2e9e5b">■ green = the two views fused / no longer separable</span>. Where a *shared* module is applied to each image in turn, the arrow is **duplicated** (one blue, one orange) to show both images traverse the same weights.

```mermaid
flowchart TB
    I1["img1 — MASKED ~90%<br/><i>the prediction target; hiding most of it forces<br/>the model to learn scene structure, not copy pixels</i>"]
    I2["img2 — full (reference)<br/><i>a second view of the same scene, kept intact to<br/>supply the cues needed to fill in img1</i>"]

    subgraph ENC["Encoder — SHARED weights (Siamese)"]
        direction TB
        PE["patch_embed<br/><i>cuts each image into fixed patches and<br/>projects them to tokens — the ViT entry point</i>"]
        EB["enc_blocks ×12<br/><i>the ViT transformer trunk that turns patch<br/>tokens into contextual visual features</i>"]
        EN["enc_norm<br/><i>final LayerNorm stabilising the encoder output</i>"]
    end

    F1["feat1 (visible patches only)<br/><i>encodes just the ~10% unmasked patches —<br/>cheap, since masked patches are skipped here</i>"]
    F2["feat2 (memory)<br/><i>full features of img2, later read by the<br/>decoder as cross-attention context</i>"]
    MT["mask_token<br/>(fills masked positions)<br/><i>a single learned vector inserted at every hidden<br/>location so the decoder has a slot to predict into</i>"]
    DE["decoder_embed (Linear enc→dec)<br/><i>projects encoder-width tokens down to the<br/>(smaller) decoder width</i>"]
    DEC["dec_blocks ×8<br/>(self-attn + cross-attn → feat2)<br/><i>reconstructs img1: self-attention reasons over its<br/>own tokens, cross-attention pulls evidence from img2</i>"]
    DN["dec_norm<br/><i>LayerNorm on decoder output before the head</i>"]
    PH["prediction_head<br/>(Linear → patch²·3)<br/><i>maps each token back to the raw RGB pixels<br/>of its patch</i>"]
    OUT["reconstructed RGB pixels of img1<br/><i>the self-supervised objective: match the<br/>hidden pixels, no labels required</i>"]

    I1 --> PE
    I2 --> PE
    PE --> EB
    PE --> EB
    EB --> EN
    EB --> EN
    EN --> F1
    EN --> F2
    F1 --> DE
    DE --> DEC
    MT --> DEC
    F2 -- "cross-attn memory" --> DEC
    DEC --> DN
    DN --> PH
    PH --> OUT

    %% blue = img1 · orange = img2 · green = fused (decoder mixes img2 memory into img1)
    linkStyle 0,2,4,6,8,9,10 stroke:#2f6fdb,stroke-width:2px
    linkStyle 1,3,5,7,11 stroke:#d9822b,stroke-width:2px
    linkStyle 12,13,14 stroke:#2e9e5b,stroke-width:2px
```

## 2. DUSt3R (`AsymmetricCroCo3DStereo`)

Same Siamese **encoder**, but now **two cross-coupled decoders** and **two
pointmap heads**. No masking, no pixel head. Both outputs are 3D pointmaps in
**view1's coordinate frame** (the asymmetry is now in the *coordinate frame*,
realised by the two separate decoder stacks).

**`*` marks parts pretrained by CroCo v2's self-supervised training** (and thus
loaded from its checkpoint — for the parts DUSt3R keeps, these two are the same
set). Unmarked parts are DUSt3R-specific and trained from scratch.

**Edge colours:** <span style="color:#2f6fdb">■ blue = view1 stream</span> · <span style="color:#d9822b">■ orange = view2 stream</span> · <span style="color:#2e9e5b">■ green = cross-coupling where the two paths exchange information</span>. The two streams never merge into one tensor — they stay separate end to end — so the only green edge is the per-layer cross-attention between the decoders. Shared modules (`decoder_embed`, `dec_norm`) get **duplicated** arrows, one per view.

```mermaid
flowchart TB
    I1["view1 img<br/><i>defines the output coordinate frame —<br/>everything is predicted relative to this view</i>"]
    I2["view2 img<br/><i>a second view of the scene; its geometry is<br/>re-expressed into view1's frame</i>"]

    subgraph ENC["Encoder — SHARED weights (Siamese)"]
        direction TB
        PE["* patch_embed<br/>(PatchEmbedDust3R / ManyAR)<br/><i>tokenises images; ManyAR variant lets a batch<br/>mix aspect ratios during training</i>"]
        EB["* enc_blocks ×24 (ViT-L, RoPE)<br/><i>deeper ViT-L trunk with rotary position encoding —<br/>the heavy feature extractor reused from CroCo</i>"]
        EN["* enc_norm<br/><i>final encoder LayerNorm</i>"]
    end

    F1["feat1 + pos1<br/><i>view1 tokens plus their positions, fed to the<br/>view1 decoder path</i>"]
    F2["feat2 + pos2<br/><i>view2 tokens plus positions, fed to the<br/>view2 decoder path</i>"]
    DE["* decoder_embed (Linear, SHARED)<br/><i>one projection (encoder→decoder width) applied<br/>to both views, so they enter a common token space</i>"]
    G1["f1<br/><i>view1 tokens in decoder space</i>"]
    G2["f2<br/><i>view2 tokens in decoder space</i>"]

    subgraph DEC["Two decoder stacks — cross-coupled every layer"]
        direction LR
        D1["* dec_blocks ×12<br/>(view1 path)<br/><i>decodes view1; cross-attends to view2 so it can<br/>place view1 geometry using both views</i>"]
        D2["* dec_blocks2 ×12<br/>(view2 path — deepcopy of dec_blocks)<br/><i>separate weights let view2 be re-expressed into<br/>view1's frame — this is where the asymmetry lives</i>"]
    end

    DN["* dec_norm (SHARED)<br/><i>one LayerNorm reused by both paths before<br/>their heads</i>"]
    H1["downstream_head1<br/>(linear / dpt)<br/><i>regresses dense 3D points for view1;<br/>dpt = higher-res, linear = lightweight</i>"]
    H2["downstream_head2<br/>(linear / dpt)<br/><i>separate head for view2's pointmap, since the<br/>two outputs live in the same but distinct mapping</i>"]
    O1["pred1: pts3d + conf<br/>(view1 frame)<br/><i>view1's pointmap with per-pixel confidence<br/>weighting the loss</i>"]
    O2["pred2: pts3d_in_other_view + conf<br/>(also view1 frame)<br/><i>view2's pointmap, already aligned into view1's<br/>frame — gives cross-view registration for free</i>"]

    I1 --> PE
    I2 --> PE
    PE --> EB
    PE --> EB
    EB --> EN
    EB --> EN
    EN --> F1
    EN --> F2
    F1 --> DE
    F2 --> DE
    DE --> G1
    DE --> G2
    G1 --> D1
    G2 --> D2
    D1 <-. "each layer:<br/>blk1 attends f1→f2,<br/>blk2 attends f2→f1" .-> D2
    D1 --> DN
    D2 --> DN
    DN --> H1
    DN --> H2
    H1 --> O1
    H2 --> O2

    %% blue = view1 · orange = view2 · green = cross-coupling exchange
    linkStyle 0,2,4,6,8,10,12,15,17,19 stroke:#2f6fdb,stroke-width:2px
    linkStyle 1,3,5,7,9,11,13,16,18,20 stroke:#d9822b,stroke-width:2px
    linkStyle 14 stroke:#2e9e5b,stroke-width:2px
```

> Note on `* dec_blocks2`: there is no `dec_blocks2` in the CroCo checkpoint —
> it is populated by **duplicating** the `dec_blocks` weights at load time
> (`load_state_dict`, `model.py:91-98`), so it still *originates* from CroCo.

## 3. Every node, side by side

One row per node appearing in either diagram. **CroCo** / **DUSt3R** columns say
whether that node is part of the forward pass in each model (✅ used · ⚪️
present-but-inactive · ❌ absent). Names in `code font` are the actual attribute
names in `croco/models/croco.py` / `dust3r/model.py`.

| Node (diagram label) | CroCo | DUSt3R | What it is and why it's there |
|---|:---:|:---:|---|
| **Input image** (`img1`/`view1`, `img2`/`view2`) | ✅ | ✅ | The two RGB images of the scene. **CroCo:** `img1` is heavily masked (the reconstruction target), `img2` is intact and serves only as a reference. **DUSt3R:** neither is masked; both are full images, and `view1` additionally *defines the output coordinate frame* — every predicted 3D point, for both views, is expressed relative to `view1`'s camera. |
| **`patch_embed`** | ✅ | ✅ | Conv/linear projection that slices each image into non-overlapping patches and embeds each as a token — the ViT entry point. CroCo uses plain `PatchEmbed`; DUSt3R swaps in `PatchEmbedDust3R` (fixed shape, inference) or `ManyAR_PatchEmbed` (lets one batch mix aspect ratios during training). Same weight dimensions, so CroCo's weights load directly. |
| **`enc_blocks`** | ✅ ×12 | ✅ ×24 | The ViT transformer trunk: stacked self-attention blocks that turn patch tokens into contextual visual features. This is the heavy feature extractor and the **main thing reused from CroCo** (ViT-L, ×24, with RoPE in the v2 backbone). Run **Siamese** — the same weights process both images, in fact as one concatenated batch (`_encode_image_pairs`). |
| **`enc_norm`** | ✅ | ✅ | Final `LayerNorm` applied to the encoder output. Shared across both images; transfers verbatim from CroCo. |
| **`feat1` / `feat2`** (encoder output) | ✅ | ✅ | The per-image encoder feature tokens. **CroCo:** `feat1` covers only the ~10% *visible* patches (masked patches are skipped here for efficiency), while `feat2` is the full feature set used as cross-attention memory. **DUSt3R:** `feat1`/`feat2` are the full token sets, each carried with its positional encoding (`pos1`/`pos2`) into its own decoder path. Not a learned module — just the activation tensors flowing between encoder and decoder. |
| **`mask_token`** | ✅ | ⚪️ | A single learned vector inserted at every *hidden* patch position so the decoder has a slot to predict into. **Essential to CroCo's masked-reconstruction objective.** DUSt3R inherits the parameter (it's defined on `CroCoNet`) but never masks anything, so it is **unused** at train and inference time — dead weight kept only because the class is subclassed. |
| **`decoder_embed`** | ✅ | ✅ | Linear projection from encoder width to the (smaller) decoder width, so tokens enter the decoder's token space. In DUSt3R it is a **single shared instance** applied to *both* view streams in turn (hence the duplicated arrows in the diagram). Transfers from CroCo. |
| **`f1` / `f2`** (decoder-space tokens) | — | ✅ | DUSt3R-only intermediate: the outputs of `decoder_embed`, i.e. `feat1`/`feat2` re-expressed in decoder width, ready to feed the two decoder stacks. (CroCo has the analogous tensor but the diagram doesn't name it separately.) Activations, not a module. |
| **`dec_blocks`** | ✅ | ✅ | The transformer decoder stack. **CroCo:** the *only* decoder — self-attention over `img1`'s tokens plus cross-attention into `feat2`, reconstructing `img1`. **DUSt3R:** this becomes the **`view1` path** (one of two stacks). Each layer does self-attention *and* cross-attention to the other view's tokens. CroCo's decoder weights initialise it. |
| **`dec_blocks2`** | ❌ | ✅ | DUSt3R's **second decoder stack**, the `view2` path. It does **not** exist in the CroCo checkpoint; it is created by `deepcopy`-ing `dec_blocks` and populated by duplicating those weights at load (`load_state_dict`, `model.py:91-98`), then diverges during training. **This second stack is where the asymmetry lives** — it re-expresses `view2`'s geometry into `view1`'s frame. Cross-coupled with `dec_blocks` every layer (the green edge). |
| **`dec_norm`** | ✅ | ✅ | `LayerNorm` on the decoder output before the head. In DUSt3R a **single shared instance** normalises both decoder paths (duplicated arrows). Transfers from CroCo. |
| **`prediction_head`** | ✅ | ❌ | CroCo's output head: a `Linear` mapping each decoder token back to its patch's raw RGB pixels (`patch²·3`) for the reconstruction loss. **DUSt3R removes it entirely** — `_set_prediction_head` is a no-op — because DUSt3R predicts geometry, not pixels. |
| **`downstream_head1` / `downstream_head2`** | ❌ | ✅ | DUSt3R's **two pointmap heads** (`linear` for the 224 model, `dpt` for higher-res 512 models). Each regresses, per pixel, a 3D point `(x,y,z)` plus a confidence scalar. Separate instances for the two views. **New to DUSt3R, trained from scratch** (no CroCo counterpart). |
| **Output** (`reconstructed RGB` / `pred1`,`pred2`) | ✅ | ✅ | **CroCo:** the reconstructed RGB pixels of the masked `img1` — the self-supervised target, no labels needed. **DUSt3R:** `pred1` = `view1`'s pointmap + confidence (in `view1` frame); `pred2` = `view2`'s pointmap + confidence, already aligned **into `view1`'s frame** (`pts3d_in_other_view`), which is what gives cross-view registration for free. |

### Key takeaways
- **Encoder (`patch_embed` → `enc_blocks` → `enc_norm`) is fully shared** between
  the two images *and* transfers verbatim from CroCo — that's why warm-starting
  from `CroCo_V2_ViTLarge_BaseDecoder.pth` works.
- **The two decoders are where the asymmetry lives.** `dec_blocks2` starts as an
  exact copy of `dec_blocks` (duplicated at checkpoint load) and then diverges
  during training; each layer they cross-attend to each other, so information
  flows between the two views at every decoder block.
- **`decoder_embed` and `dec_norm` are shared** across both decoder paths; only
  the decoder blocks and the output heads are duplicated.
- **What changes from CroCo to DUSt3R is only the ends:** masking and the
  pixel `prediction_head` are dropped (and `mask_token` goes unused), and two
  `downstream_head`s predict pointmaps instead. Everything in between is reused.
