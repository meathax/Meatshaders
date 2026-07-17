# Port design: CRT-Guest-Advanced / CRT-Royale / CRT-Royale-Kurozumi → MiSTer (1080p)

Date: 2026-07-17. Targets produced by validated reference modules in the session
scratchpad `targets/` dir (guest_advanced_ref.py, royale_ref.py, kurozumi_ref.py,
each with transfer / beam_weight / ref_vertical / h_kernel / mask_spec / notes and
JSON grids; sources pinned to libretro/slang-shaders 3b0d6aa1d134a168478cd9c904a866d969f8882b).
Hardware arithmetic verified against RTL (see mister_model.py header, F1–F5).

## Architecture (all three shaders)

1. **Gamma LUT = beam-center transfer.** `LUT(x) = round(255 * transfer(x))`
   per channel (Kurozumi: genuinely 3-channel — grade curve incl. warm flare
   black lift). This makes the V-row target `R(f,x) = ref_vertical(f,x) /
   transfer(x)` equal 1.0 at every scanline peak, so **all V rows are ≤ 256 by
   construction** — no scaler clipping anywhere (Easymode lesson D6), and the
   "late gain" lives in the LUT where it cannot clip structure. Royale's ~1.2%
   brightpass dip near x≈0.83 is smoothed to keep the LUT monotone.

2. **Adaptive V filter.** Per phase p (f = p/256), tap distances
   d = [f+1, f, 1−f, 2−f]. Endpoint row *shapes* ∝ beam_weight(d, L) at L=0⁺
   (set A, lum=0) and L=1 (set B, lum=255); endpoint row *sums* fitted by
   least squares so that the hardware blend (exact F2 truncation) at
   lum = LUT(x) reproduces 256·R(f,x) over the full x grid. Sums capped at 256.
   Quantized with exact conjugate symmetry (quantize.py).

3. **Fixed V fallback** fitted the same way but single-set, minimizing error
   over the x grid (best constant-profile compromise) — used by the
   "Fixed Compatibility" presets and as graceful behavior for v6 cores
   (RTL fact F5: v6 cores silently get only set A of adaptive files; set A =
   dark endpoint is NOT self-sufficient for Royale/Kurozumi, so compat presets
   must exist and the README must say so).

4. **H filter** = h_kernel(f) folded to 4 taps, DC = 256 at every phase.
   Guest (2-tap + small negative lobes) and Royale (Quilez smootherstep) fit
   exactly; Royale blends 92.5% Quilez + 7.5% diffusion Gaussian (σ≈1.45 src
   px) truncated to the 4-tap window (residual ~2.5% of the veil documented);
   Kurozumi is a native Gaussian σ=0.32 fit.

5. **Interlace (ifilter)** = dedicated "No Scanlines" table: bright-endpoint
   beam shape normalized to DC 256 (the shader's own vertical interpolation
   with scan modulation removed). Every preset names it explicitly (v4 lesson).

6. **Masks** (encoded-space matching: linear m → token step round(16·m^(1/2.2))
   — exact for power-law encodes):
   - Guest: CGWG magenta/green 2×1: linear (1, 0.7, 1)/(0.7, 1, 0.7);
     0.7^(1/2.4) = 0.862 → 14/16. Tokens `50e,20e` (X bitmask: 4=R,2=G,1=B).
   - Royale: shader renders a 24×24 slot tile at 1080p but MiSTer masks are
     hardware-limited to 16×16 (shadowmask.sv 4-bit indices) → extract the
     minimal vertical/horizontal period from the computed tile and fit tokens
     by least squares; report the approximation error. Net multipliers up to
     1.78 encoded are representable ((16+Y)/16 ≤ 1.9375).
   - Kurozumi: shader itself collapses the grille to a 2-px R/B alternation at
     1080p: target encoded (1.341, 1.009, 0.546)/(0.546, 1.009, 1.341). Token
     model can't give G its own value (one shared "other" nibble) → simulate
     candidates (Y=5 with Z=15 vs Z=14; plus no-mask fallback) with the exact
     F3 truncation model and pick minimum error on gray ramps + primaries;
     ship runner-up as an alternative mask. Avoid Z=15 if error is close
     (worst truncation noise, RTL D3).

7. **Moiré guard** (Easymode lesson D4): every generated V table is simulated
   at 4.5× and 4.821× flat fields across the gray ramp; the per-period
   trough-stddev metric must be ≤ ~0.03 of full scale. Kurozumi's deep
   constant scanlines are expected to violate this → primary preset stays
   exact ("pixel perfect" goal) with README recommending vscale_mode=1;
   an additional "Anti-Moire" variant widens the beam by the minimal factor
   that passes, clearly labeled as display-tuned deviation.

8. **Presets per shader**: Default (adaptive), Fixed Compatibility,
   No Gamma (gamma=off, documented approximation; gamma support is
   core-dependent), TATE (slots swapped, adaptive file in the H slot — only
   one adaptive slot exists, F2 — explicit ifilter, mask `1x rotated` where
   the pattern is directional), Kurozumi Anti-Moire. maskmode always explicit.

## Naming

Files: `CRT Guest Advanced (Port)_H.txt`, `..._V Adaptive.txt`, `..._V Fixed.txt`,
`..._V No Scanlines.txt`; same pattern for `CRT Royale (Port)` and
`CRT Royale Kurozumi (Port)`. Gamma: `CRT <name> (Port).txt`. Masks:
`CRT <name> <pattern> (Port).txt`. Presets: `CRT <Name> - <Variant>.ini`.
No collisions with the existing v4 pack or the official MiSTer repos
(prior-art scan found no existing ports of these shaders).

## v5.1 revisions (2026-07-17, second pass)

Five changes, each driven by a measurement that contradicted an assumption above.

1. **The acceptance metric was wrong.** Mask-off RMSE silently assumes the
   port's mask equals the shader's, so it cannot see clamp-ORDER error (the
   shader clips `clamp(B*m)` after the mask; MiSTer clamps the V stage then
   saturates in the mask) and it is actively backwards for a gain-split build.
   `rmse_exact_masked` compares port pixels against shader pixels through the
   mask and is now the gate. It is roll-invariant: a mask tile shifted a few
   output pixels is the same mask to the eye.

2. **Gain-split factorization** (`fit_gain_split` / `build_lut_gain` /
   `fit_v_adaptive_gain`). The LUT is both the tone curve *and* the adaptive
   control, so wherever `transfer()` clips, the LUT pins at 255, the control
   pins with it, and every input in that band gets identical V rows while the
   real trough still varies — information destroyed before the V stage. Fix:
   the LUT carries the PRE-clip beam `B(x)/G` (strictly increasing, so the
   control keeps resolving) and the mask carries `G`. Saturation then happens
   at the mask stage, which is where the shader clips too. `G` is bounded by
   the *dark* mask multipliers (shared `Z/16 <= 0.9375` nibble), not the lit
   ones. Requires `ref.transfer_unclipped` / `ref.ref_vertical_unclipped`.
   Easymode: G=1.105 → 3.16 codes vs 8.37 baseline and 4.97 for v4.
   This is what v4's Lottes 47e mask was doing empirically; it now falls out
   of the math (the search re-derives 42f for Easymode and the 47e family for
   Lottes unprompted).

3. **Model-in-the-loop LUT refinement** (`refine_lut_exact`, and
   `refine_lut_masked` for the gain-split path — using the mask-off variant on
   a gain-split build undoes the split). MiSTer's FIR truncates, so a unity-DC
   row returns g-1: every level sat 1–4 codes under its ideal value because the
   fit ran in ideal arithmetic. Each code's LUT entry only affects that code, so
   the search is separable and cheap.

4. **Fixed fallback gets its own LUT** (`fit_v_fixed_paired`). Without an
   adaptive control the model collapses to `out = u(x)*S(f)/256` — a rank-1
   factorization whose optimal warp differs from the adaptive one. Alternating
   least squares on (u, S) instead of averaging per-x ideal rows: Guest 8.7→4.5,
   Royale 38.2→23.2, Kurozumi 26.4→17.5.

5. **Mask pipeline** — three real defects:
   - `minimal_period_tile` averaged the repeats, which is L2-optimal and
     therefore a trap: a component ANTISYMMETRIC under the chosen period is
     cancelled to zero and the RMSE *improves* for destroying it. Royale's
     24x24 slot tile is antisymmetric under y→y+12, so the shipped 12x3 mean
     was a plain aperture grille with **0% of the slot signal**. Now
     `choose_mask_tile` scores per-frequency spectral overlap and offers
     verbatim slices (Royale ships a 12x6 verbatim slice).
   - The encode exponent was hardcoded to 2.2; Kurozumi's `lcd_gamma` is 2.4
     (`_output_gamma` reads it from the reference).
   - A v2 token gives its two non-selected channels ONE shared value, so
     Kurozumi's (1.30, 1.01, 0.57) is unrepresentable and a single-token fit
     spends blue to buy luma, flattening the R/B ripple. `fit_mask_column_dither`
     spreads it across two HORIZONTAL slots (exhaustive pair search via a Gram
     matrix — shortlisting by solo error discards exactly the straddling pairs
     that dither well). Partners sit one period apart so the base alternation
     survives and the residual moves to a higher frequency; horizontal dithering
     adds no vertical structure, unlike a 2-row tile which would beat against
     the 4.5x scanline. Exact error 11.6 → 3.0 codes.
   Token scoring now runs through `mister_model.mask_multiply` (per-set-bit
   truncated shifts) rather than ideal m/16 — m=15/16 runs ~1.5 codes low while
   m=16/16 is exact, which flips winners.

**Lottes is deliberately NOT rebuilt.** Measured on the same metric, the shipped
v4 scores 1.233 against a best-achievable 1.208 — a 2% difference on the user's
gold standard. v4's gamma=off + gain-in-mask factorization already mirrors the
shader's clamp order. Verified, left alone.

## Validation gates (task 8)

- fileio validators pass on every generated file; presets resolve case-sensitively.
- End-to-end model simulation (mister_model.py, exact RTL arithmetic) vs
  reference module over the f × x grid: report RMSE in output codes per shader
  (target: comparable to v4's Lottes ≈ 1.8–4 codes; document where and why).
- Flat-field brightness parity vs reference at x = 0.25/0.5/0.75/1.0.
- Moiré metrics at 4.5×/4.821× for every V table.
- Clipping audit: no V row sum > 256 anywhere.
