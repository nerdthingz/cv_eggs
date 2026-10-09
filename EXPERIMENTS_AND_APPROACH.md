# Approach and experiments

## 1. Setup
- **Task 1 (reference masks):** SAM 3 (`facebook/sam3`) with the text prompt "egg" (fallbacks: "white egg", "brown egg", "an egg in a hand"; score threshold 0.5, then 0.25). The union of all egg masks is the ground truth. SAM found an egg in 109 of 110 images; the one without is not scored.
- **Split (Task 2):** 88 dev / 22 eval (4:1). Images are grouped by original photo name (text before `.rf.`) and near-duplicates are merged by perceptual hash (also checks mirrored copies). Whole groups go to one side, so no leakage. Eval was used once, after freezing the parameters.
- **Metric:** IoU against the SAM mask at 400 px (long side). On 87 dev images, differences below ~0.02 are noise.
- **Key fact:** an ellipse fitted to a SAM egg mask matches it at 0.968 IoU, so the job is to find the right ellipse.

## 2. Final algorithm (no deep learning)
1. **Clean:** denoise + contrast boost (CLAHE on L).
2. **Regions:** k-means (k=12) on Lab colour, split into connected pieces; Canny edges act as walls between pieces.
3. **Egg colour per image:** 25 seed points (5x5 grid). Each seed region's mean colour is the egg-colour guess; every region within a Lab distance (10/18/28 x 1.30) is merged. Morphological opening (5/15/31) cuts thin bridges to fingers.
4. **Ellipse + score:** fit an ellipse to each blob (enlarged x1.147). Score = ellipse fit x egg-size prior x edge support under the outline (power 1.85) x centre prior. Best score = the egg.
5. **More eggs:** extra blobs are kept if they are ellipse-like (>= 0.85), overlap little (<= 0.2) and score >= 0.7x the best (scored without the centre prior). Max 4.
6. **GrabCut** seeded with each ellipse snaps the outline to the real boundary.

## 3. Experiments (dev split, mean IoU)
| # | Trial (what and why) | Key parameters | Dev IoU | Decision |
|---|---|---|---|---|
| 1 | Candidate regions (Otsu, k-means, edges) scored by shape / colour / contrast | ~22 candidates per image | oracle 0.62; best pick 0.49 (shape 0.45, colour 0.20, contrast 0.33) | Shape strongest, colour weakest. Ceiling too low |
| 2 | Sigmoid: keep all candidates above a threshold (multi-egg) | thr 0.3-0.9 | 0.19-0.30 | Discarded: false eggs |
| 3 | Direct ellipse search on edge alignment (~400k ellipses) + centre prior | area 3-25%, aspect 0.55-0.95 | 0.33, 0.43 with centre prior; oracle 0.62 | Discarded: same ceiling. Skin penalty hurt (0.42), contrast gave nothing |
| 4 | GrabCut on top of 1 and 3 | 4 iterations | 0.50 / 0.49 | Small gain, idea kept |
| 5 | Keep every white/brown pixel, then fit ellipse | HSV ranges | 0.25 (37% no egg found) | Discarded: hand and wall share the colours, 72% of kept pixels were not egg |
| 6 | Per-image colour model: colour of a seed patch, keep close pixels | Lab thr 12/20/30 | 0.42 | Idea kept. Masks about 27% too small |
| 7 | Stacked: clean, k-means regions with edge walls, merge by seed-region colour, ellipse | k=8, thr 10/18/28, grow 1.1 | 0.49 | Kept |
| 8 | 4x4 overlapping windows | window 0.5 | 0.50 | Discarded: no gain, 17x slower |
| 9 | + edge-support term in the score | power 1 | 0.52 | Kept |
| 10 | + GrabCut from the ellipse | 4 iterations | 0.56 | Kept: biggest single gain |
| 11 | Optuna, 10 TPE trials | k=12, grow 1.147, edge power 1.85, thr scale 1.30 | 0.58 | Kept. Best values at range edges and tuned on dev, so slightly optimistic |
| 12 | Egg-aspect prior; smoothing (hull + blur, or ellipse fit of the mask) | - | 0.591 / 0.581 / 0.578 | Discarded: within noise, outlines already smooth |
| 13 | Multi-egg: 5x5 seeds, extra eggs scored without the centre prior, ratio 0.7 / 0.5 / 0.35 | - | 0.598 / 0.597 / 0.596 (multi-egg IoU 0.54 / 0.57 / 0.67; false extra eggs 8 / 12 / 24) | 0.7 chosen |

**What the experiments showed**
- Methods 1 and 3 both topped out at an oracle of 0.62, so finding good candidates mattered more than scoring them.
- A fixed colour range fails (skin, walls, eggs overlap). A per-image colour model with whole-region merging fixed it.
- Edge support and GrabCut improved picking and boundaries. Tiling, smoothing and priors added nothing and were dropped.
- Remaining failures: low-contrast eggs (egg colour close to hand or background) and eggs found too small.

## 4. Video (Task 4)
- **Detection:** the same algorithm every 2nd frame, adapted: seeds on a 7x7 grid over the whole frame, no centre prior, looser size prior, extra eggs need >= 0.4x the best score and ellipse fit >= 0.8.
- **Tracking:** a Kalman filter (constant velocity) predicts each egg's centre. Hungarian matching assigns detections using centre distance (in egg diameters, gate 1.5) plus change in size. Unmatched detections start a new ID, tracks survive 10 missed detections, and an ID is drawn only after 3 matches.
- No ground truth exists for the video, so it is judged visually.

## 5. AI tools disclosure
An AI assistant (Claude) was used to discuss approaches and draft code. Every method above was run, checked and can be explained by the author.
