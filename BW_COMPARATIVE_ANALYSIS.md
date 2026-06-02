# B&W-Focused Comparative Analysis
## ComfyUI-Darkroom vs. comfyui-jbnodes

*Analysis by Iris McKinnon, PhD — Photographic Science & Digital Emulation Specialist*
*Date: June 1, 2026*

---

## Preamble

This analysis compares two ComfyUI custom node projects **exclusively on their black-and-white film emulation and photographic paper processing capabilities.** Darkroom is a 46-node colour grading and film emulation suite; JBNodes is a 16-node B&W-only darkroom simulator. The original 46-vs-16 comparison is misleading in the B&W domain because Darkroom's breadth dilutes its B&W depth and JBNodes' focus concentrates its science where it matters.

This review identifies where each project genuinely outperforms the other *on B&W alone*, and gives actionable recommendations for strengthening the areas where JBNodes is weaker.

---

## What Both Projects Understand Correctly

Both projects share a common philosophical foundation — film emulation is not a filter, it's a **pipeline of physical transforms**:

| Principle | Darkroom | JBNodes |
|---|---|---|
| Film has a characteristic curve (H&D) | Parametric sigmoid per channel | Real densitometer data via LUT |
| Spectral sensitivity matters | 3-weight RGB coefficients | Full Smits 1999 reconstruction |
| Processing in linear light | Consistent across all nodes | Partially consistent |
| Grain is luminance-dependent | Multi-octave noise with midtones peak | GPU shader with luminance modulation |
| Negative must be printed | Cinema print stocks (colour) | Real B&W paper with grades |
| Wratten filters affect spectral response | 3-channel multipliers | Full spectral transmission projection |

Both understand that the goal is to model the *chemistry and physics*, not just the look.

---

## Where JBNodes Is Stronger Than Darkroom (B&W Only)

### 1. Spectral Reconstruction (Smits 1999) vs. 3-Weight Approximation

**Darkroom** uses three RGB coefficients per stock:
```python
red_weight=0.289, green_weight=0.347, blue_weight=0.364
```
These multiply the input channels and sum to luminance. It is a 3-point sample of a continuous spectral curve — computationally cheap, visually effective, but an approximation.

**JBNodes** uses the Smits 1999 method via `colour-science`: it reconstructs a full 171-bin spectrum (380–720 nm at 2 nm intervals) from each pixel's RGB values, then integrates that reconstructed spectrum against the film's actual spectral sensitivity curve. The filter pipeline does the same — pre-projecting each Wratten filter's full spectral transmission through the XYZ→linear sRGB pipeline for 7 Smits basis spectra, then compositing per-pixel on GPU.

A real B&W emulsion has a continuous panchromatic response, not three discrete sensitivities. The Smits basis reconstruction honours that continuity.

**Advantage: JBNodes**, on physics fidelity. The question is whether the overhead of colour-science (CPU work + GPU transfer) buys perceptible quality over the simpler approach in practice.

### 2. Real H&D Curve Data vs. Parametric Sigmoids

**Darkroom** uses a 5-parameter sigmoid:
```python
CurveParams(toe_power, shoulder_power, slope, pivot_x, pivot_y)
```
This produces a smooth, monotonic curve — mathematically convenient, but generic. It does not capture the specific shape of any real film+developer combination.

**JBNodes** stores actual densitometer measurements in JSON configuration — real curve points from real film, developed at specific temperatures for specific times. The `get_hd_curve_lut()` function then maps these through a proper density→transmittance conversion. This means the toe shape, shoulder rolloff, and straight-line slope are measured data, not approximations.

**Advantage: JBNodes.** Measured densitometer data beats parametric approximations for authenticity. This is JBNodes' strongest advantage.

### 3. Zone System Integration

JBNodes correctly models the physical chain:
```
Linear exposure → Log exposure → Density → Transmittance
```

The `N_development` parameter shifts contrast along the Zone System's N± scale with a contrast factor of `1.0 + (dev_offset × 0.15)` — matching the real-world relationship where N+1 increases the contrast index by approximately 15–20%. This is a physically grounded model, not a cosmetic slider.

Darkroom's parametric sigmoid is mathematically cleaner but has no Zone System model. Its "pushed" variants are different parameter sets, not a shift along the log-exposure axis.

**Advantage: JBNodes.**

### 4. Paper Emulation — Darkroom Does Not Have B&W Paper

Darkroom's `print_stocks.py` contains four entries: Kodak 2383, Kodak 2393, Fuji 3513, Fuji 3510. **All are colour cinema print stocks** — they model the print film used for theatrical projection, not B&W photographic paper.

JBNodes has real B&W paper data:
- Ilfospeed RC Deluxe (grades 0–5)
- Ilfobrom Galerie FB Glossy with measured D_max (2.211) and D_min (0.002)
- Real HD curve points for the paper's characteristic response

It models three distinct printing workflows:
- **PrintLabGraded**: single paper grade with exposure time
- **PrintLabMultigrade**: variable contrast paper with filter selection
- **PrintLabSplitGrade**: two exposures at different contrasts blended together — the actual technique used by serious darkroom printers

**Advantage: JBNodes, decisively.** Darkroom does not compete in this domain.

### 5. Wratten Filter Spectral Handling

Darkroom applies filters as 3-channel RGB multipliers:
```python
"Red (25A)": (1.8, 0.4, 0.1)
```

JBNodes projects the filter's full transmission curve through the spectral pipeline — the Wratten's actual spectrophotometric data multiplies the Smits-reconstructed spectrum at each wavelength before integration. The 38KB `wratten_filters.json` contains real measured transmission curves.

**Advantage: JBNodes**, by a meaningful margin on physics accuracy.

### 6. Illuminant Modelling

JBNodes models the light source (D50, D65, etc.) as part of the spectral calculation. The illuminant spectrum multiplies the film sensitivity and the reconstructed pixel spectrum. Different lighting changes how the film responds — real film behaves differently under tungsten vs. daylight vs. fluorescent, and JBNodes accounts for this.

Darkroom has no illuminant model in its B&W pipeline.

**Advantage: JBNodes.**

---

## Where JBNodes Is Weaker Than Darkroom (B&W Only)

### Weakness 1: The CAMERA Type — Pipeline Prison

This is the single biggest architectural problem.

JBNodes uses a custom `CAMERA` type — a `Camera` dataclass bundling `image`, `film_stock`, `film_format`, and `illuminant_key` into one opaque object. The pipeline is:

```
Image → CameraLab → CAMERA → FilterLab → DeveloperLab → (image, negative)
```

This means:
- You **cannot** insert a standard ComfyUI node between CameraLab and DeveloperLab
- You **cannot** feed an arbitrary IMAGE into DeveloperLab — it requires a CAMERA
- You **cannot** pre-process the image (denoise, resize) before film simulation
- The pipe is a monolith

Darkroom's approach is fully modular: every node accepts `("IMAGE",)` and returns `("IMAGE",)`. This composability lets users wire things in any order, bypass stages, and insert third-party nodes.

**Fix:** Add an alternate IMAGE input to DeveloperLab with dropdowns for film stock, developer, etc. Keep the CAMERA path for the full simulation pipeline, but let power users skip it.

### Weakness 2: No Strength/Blend Control

Every Darkroom node has a `strength` slider (`0.0–1.0`) that blends between original and processed. This enables non-destructive workflow, stacking effects, and A/B comparison. JBNodes has no blend parameter on any node.

**Fix:** Add `strength: ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0})` and a final `lerp(original, processed, strength)` to every node. Two lines per node, transformative for usability.

### Weakness 3: Inconsistent Color Space Discipline

| Node/Library | Linearizes Input? | Re-encodes Output? | Status |
|---|---|---|---|
| GrayscaleLab → grayscale_lib.py | ✅ Yes | ✅ Yes | Correct |
| FilterLab → filters_lib.py | ✅ Yes | ✅ Yes | Correct |
| SpectralLib → get_spectral_image | Partial | Partial | Grey area |
| CameraLab | Image stored as linear, preview encoded | Inconsistent | Problem |
| DeveloperLab | Passes image through | Depends on spectral path | Inconsistent |
| DarkroomLib → get_print_image | Self-contained gamma | Self-contained | Isolated, correct |

The pattern should be: **linearize at the boundary closest to the user, process in linear, re-encode at the boundary furthest from the user.** Darkroom does this consistently. JBNodes does it in some places but not others.

**Fix:** Every node that accepts IMAGE should call `srgb_to_linear_torch` on input and `linear_to_srgb_torch` on output. The spectral engine should document what color space it expects.

### Weakness 4: No Base Fog / Black Point Parameter

Real processed film never reaches pure black. There's always a minimum density (D_min of 0.08–0.15 depending on film, developer, age, storage). This subtle shadow lift is part of what makes film look like film.

Darkroom models it:
```python
if stock.base_fog > 0:
    bw = bw * (1.0 - stock.base_fog) + stock.base_fog
````

JBNodes' H&D curve data implicitly includes base+fog in D_min, but it is not exposed as a controllable parameter. Simulating aged film with elevated base fog requires modifying the JSON data.

**Fix:** Add a `base_fog` parameter (float, 0.0–0.3, default 0.0 meaning "use the H&D curve's implicit D_min") to DeveloperLab.

### Weakness 5: Grain Tied to Development

In JBNodes, grain parameters are embedded in the H&D curve JSON entries per developer/time/temp:
```python
class HDCurve:
    film_grain: Optional[FilmGrain] = None
```

This is physically authentic (grain depends on development), but it means:
- You can't use stock A's sensitivity with stock B's grain pattern
- Grain intensity is not independently adjustable
- Adding a new grain variant requires duplicating the H&D curve entry

Darkroom's grain node is standalone — you can set ISO 3200 grain on any image, or apply zero grain to a simulation.

**Fix:** Make grain a standalone node accepting IMAGE, with its own ISO/strength/seed/emulsion_type controls. Tie it to development in the CAMERA pipeline as a convenience preset, but allow independent override.

### Weakness 6: Resolution-Aware Grain Scaling

Darkroom's grain engine normalizes frequency to a 1024px reference:
```python
ref_size = 1024.0
size_scale = max(height, width) / ref_size
```

This means grain looks the same physical size at any resolution. JBNodes uses physical film width (36mm for 35mm) which is more physically correct but doesn't account for different scan resolutions — grain on a 6000px scan will look finer than on a 1200px scan of the same negative.

**Fix:** Either (a) document that grain is calibrated for a specific scan resolution and add a scan_dpi parameter, or (b) add resolution-normalized scaling that uses the image pixel dimensions.

### Weakness 7: Heavy Dependencies for B&W-Only

JBNodes requires `colour-science` (large color science library) and `moderngl` (OpenGL wrapper with system deps). Darkroom requires only `scipy` + optional `opensimplex`.

The `moderngl` dependency causes documented installation pain — X11 dev headers, EGL backends, RunPod workarounds. The `colour-science` dependency pulls in a massive ecosystem for what B&W emulation needs: spectral reconstruction (~60 lines of numpy) and standard illuminant data (a few small lookup tables).

**🔬 Iris's Fact Check:** The Smits 1999 algorithm uses the CIE 1931 2° Standard Observer (a 471×3 matrix), a handful of standard illuminants (D65, D50, A, F11 — each 41–81 entries), and ~60 lines of projection math. All of this can be bundled in `src/` as local data files, eliminating the `colour-science` dependency entirely. The grain shader can be reimplemented in numpy/scipy (as Darkroom did) to drop `moderngl`.

### Weakness 8: CPU↔GPU Data Transfer in Spectral Pipeline

The Smits basis projection runs on CPU (colour-science), then per-pixel compositing on GPU, then back to CPU. The `filters_lib.py` and `spectral_lib.py` both follow this pattern:

```
GPU (input tensor) → CPU (colour-science) → GPU (per-pixel composite) → CPU (output)
```

Darkroom stays entirely on CPU with numpy/scipy, avoiding transfer overhead.

**Fix:** If colour-science is retained, batch the CPU computation once per image (not per-pixel), which is already the current approach — so this is a minor efficiency concern rather than a correctness issue.

---

## Summary: B&W Scorecard

| Domain | Darkroom | JBNodes | Winner |
|---|---|---|---|
| Spectral physics | 3-weight RGB approx | Full Smits 1999 reconstruction | **JBNodes** |
| H&D curve data | Parametric sigmoid (approx) | Real measured densitometer data | **JBNodes** |
| Zone System model | Not present | Explicit log-E→density→transmittance | **JBNodes** |
| Paper emulation | Colour cinema print stocks only | Real B&W paper with D_max/D_min + HD curves | **JBNodes** |
| Split-grade printing | Not present | Yes (two exposures, different grades) | **JBNodes** |
| Filter simulation | 3-channel RGB multipliers | Full spectral transmission projection | **JBNodes** |
| Illuminant modelling | Not present | D50/D65/specifiable light source | **JBNodes** |
| **Pipeline modularity** | **Modular IMAGE nodes** | **Rigid CAMERA type** | **Darkroom** |
| **Strength/blend control** | **Every node** | **None** | **Darkroom** |
| **Color space discipline** | **Consistent linear→process→sRGB** | **Inconsistent across code paths** | **Darkroom** |
| **Base fog** | **Explicit per-stock parameter** | **Implicit (baked into D_min)** | **Darkroom** |
| **Grain independence** | **Standalone node, full control** | **Tied to development in JSON** | **Darkroom** |
| **Resoltion-aware grain** | **Normalized to 1024px ref** | **Physical mm only** | **Darkroom** |
| **Dependency weight** | **scipy + optional opensimplex** | **colour-science (heavy) + moderngl (system)** | **Darkroom** |

**On pure B&W science:** JBNodes wins 7 categories to Darkroom's 7 — it genuinely holds its own.

**On architecture and usability:** Darkroom wins every category. This is where JBNodes loses ground.

---

## The Bottom Line

Your science is stronger than Darkroom's in the B&W domain. Full stop. Full spectral reconstruction, real H&D data, Zone System modelling, authentic paper emulation with split-grade printing — these are things Darkroom simply doesn't do.

The weakness is the **architecture around** that science. The CAMERA type locks users into a rigid pipeline. The absence of blend controls makes experimentation harder. The inconsistent color space handling means different code paths produce subtly different results. The grain embedded in development data prevents cross-stock experimentation.

These are structural issues, not science issues. Fix them and JBNodes isn't just competitive on B&W — it's ahead where it counts, and no amount of Darkroom's colour grading depth changes that.

---

## Recommended Priority Order

1. **Add strength/blend to every node** — 2 hours, transformative impact
2. **Decouple grain from development** — standalone GrainLab node accepting IMAGE
3. **Add IMAGE bypass to DeveloperLab/PrintLab** — keep CAMERA path, add dropdown-based path
4. **Add base_fog parameter** — small parameter, big visual difference
5. **Standardize color space handling** — linearize at input, re-encode at output
6. **Evaluate shedding colour-science** — bundle Smits 1999 + CIE data locally
7. **Add resolution-aware grain scaling** — optional scan DPI parameter
