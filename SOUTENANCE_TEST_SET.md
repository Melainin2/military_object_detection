# SOUTENANCE TEST SET — Curated Defense Demo Images

## Selection Criteria
- Military: ViT score > 0.05 (passes gatekeeper), diverse categories
- Civilian: ViT score < 0.05 (correctly rejected), diverse categories
- Edge cases: Challenge scenarios for discussion
- Visual impact: YOLO bounding boxes preferred for military

## Military — CORRECT DETECTION (10 images)

| # | File | Category | ViT Score | YOLO Boxes | Notes |
|---|------|----------|-----------|------------|-------|
| 1 | `mil_f35_jet.jpg` | Fighter jet | 0.8941 | ✓ 1 box | BEST DEMO — F-35, strong score + box |
| 2 | `mil_f35_01.jpg` | Fighter jet | 0.8941 | ✓ 1 box | F-35 alternative angle |
| 3 | `mil_soldier_01.jpg` | Soldier | 0.8941 | ✓ 1 box | Soldier with gear, strong visual |
| 4 | `mil_soldiers_heli_01.jpg` | Soldiers + Helicopter | 0.9664 | ✓ 1 box | Multi-object scene |
| 5 | `mil_stryker_01.jpg` | Armored vehicle | 0.9503 | ✓ 1 box | Stryker APC, strong score |
| 6 | `mil_warship_osprey_01.jpg` | Warship | 0.9917 | ✓ 1 box | HIGHEST SCORE + box |
| 7 | `mil_tanks_allied_01.jpg` | Tanks | 0.6517 | ✗ | Multiple tanks, ViT-only |
| 8 | `mil_tank_leopard.jpg` | Tank | 0.2694 | ✓ 1 box | Lower score but YOLO finds it |
| 9 | `mil_apache_01.jpg` | Helicopter | 0.2333 | ✗ | Apache, ViT-only, no box |
| 10 | `mil_warship_01.jpg` | Warship | 0.3825 | ✗ | Warship, ViT-only |

## Civilian — CORRECT REJECTION (10 images)

| # | File | Category | ViT Score | Notes |
|---|------|----------|-----------|-------|
| 1 | `civ_landscape_001.jpg` | Landscape | < 0.05 | Strong civilian |
| 2 | `civ_cat_000.jpg` | Cat | < 0.05 | Animal |
| 3 | `civ_dog_000.jpg` | Dog | < 0.05 | Animal |
| 4 | `civ_child_000.jpg` | Child | < 0.05 | Person (civilian) |
| 5 | `civ_woman_000.jpg` | Woman | < 0.05 | Person (civilian) |
| 6 | `civ_man_000.jpg` | Man | < 0.05 | Person (civilian) |
| 7 | `civ_airliner_000.jpg` | Civilian aircraft | < 0.05 | Airliner vs military jet test |
| 8 | `civ_car_000.jpg` | Car | < 0.05 | Civilian vehicle |
| 9 | `civ_ship_000.jpg` | Civilian ship | < 0.05 | Civilian vs warship test |
| 10 | `civ_food_000.jpg` | Food | < 0.05 | Unrelated category |

## Soutenance Edge Cases (11 images)

| # | File | Expected | ViT Score | Notes |
|---|------|----------|-----------|-------|
| 1 | `sout_blurred_soldier.jpg` | Military | varies | Low quality military |
| 2 | `sout_civilian_airliner.jpg` | Civilian | < 0.05 | Airliner (should reject) |
| 3 | `sout_civilian_ship.jpg` | Civilian (FP) | > 0.05 | Cruise ship → FALSE POSITIVE (YOLO sees box) |
| 4 | `sout_hunter_rifle.jpg` | Civilian | varies | Hunter with rifle (confuser) |
| 5 | `sout_low_light.jpg` | Military | varies | Low light military scene |
| 6 | `sout_military_parade.jpg` | Military | varies | Parade with military vehicles |
| 7 | `sout_military_poster.jpg` | Military | varies | Poster/art (not real) |
| 8 | `sout_police_officer.jpg` | Civilian | < 0.05 | Police vs military test |
| 9 | `sout_toy_tank.jpg` | Civilian | varies | Toy vs real test |
| 10 | `sout_videogame.jpg` | Civilian | varies | Game screenshot |
| 11 | `sout_woman_traditional.jpg` | Civilian | < 0.05 | Woman in dress |

## Primary Demo Flow (5 images, 30 seconds)

For the live demo, present these 5 images in order:

1. **`mil_f35_jet.jpg`** → "Military objects detected" with bounding box
   - Strongest case: F-35 jet, YOLO detects military_aircraft at 95.7%
   - Expected result: message + 1 detection + image_url with box

2. **`civ_landscape_001.jpg`** → "No military object detected"
   - Clear civilian: landscape, no military objects
   - Expected result: No military message + empty detections

3. **`mil_stryker_01.jpg`** → "Military objects detected" with bounding box
   - Stryker APCs: military vehicle visible
   - Expected result: military message + 1 YOLO detection

4. **`civ_cat_000.jpg`** → "No military object detected"
   - Cat picture: obvious civilian
   - Expected result: No military message

5. **`mil_warship_osprey_01.jpg`** → "Military objects detected" with bounding box
   - Warship with Osprey: HIGHEST ViT score (0.9917)
   - Expected result: military message + 1 detection

## Backup Images
- If `mil_f35_jet.jpg` runs slow: use `mil_soldier_01.jpg` instead
- If `mil_stryker_01.jpg` fails: use `mil_soldiers_heli_01.jpg`
- Any civilian image works — they all produce "No military object detected"

## Known Risks
1. `sout_civilian_ship.jpg` (cruise ship) produces FALSE POSITIVE — do NOT show unless discussing limitations
2. All Abrams tank images (mil_abrams_*.jpg) are FALSE NEGATIVES — ViT score < 0.05 — do NOT show as military
3. Images take ~3s each due to platform overhead — budget 15-20s for 5-image demo
