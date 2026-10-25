# FAO/INFOODS density mapping analysis (Food-101 -> FAO) - CONSERVATIVE

Total Food-101 labels: 101
Total valid FAO density entries parsed: 213

## Range handling
- Excel ranges like `1.01-1.03` are parsed by taking the arithmetic mean (e.g. 1.02).
- The `raw` string and `parse_kind` ('range_avg') are preserved for traceability.
- Units are g/ml = g/cm^3 (no conversion needed).
- Only positive finite densities are accepted; unparseable/non-positive entries are ignored (no guessing).
- If a range cannot be parsed, the entry is NOT used (no fallback guess).

## Matching strategy (proposed - CONSERVATIVE)
1. Normalize both sides: lowercase, replace non-alphanumeric with spaces, collapse whitespace.
2. Only accept exact normalized string match OR explicitly curated high-confidence mappings.
3. Do NOT do broad substring matching (too error-prone for composite dishes).
4. Ambiguous if multiple different FAO entries would match with similar specificity - no automatic selection; requires explicit mapping or leave unmatched.
5. NO guessed densities for unmatched foods - they will be unavailable in the density lookup.

- Exact/single high-confidence match: 2
- Ambiguous (multiple candidates): 0
- Unmatched (no candidates): 99

### EXACT/SINGLE HIGH-CONFIDENCE MATCHES
- ice_cream (norm='ice cream') <- FAO: 'Ice cream' | 0.554 g/cm^3 | raw='0.554' | kind=numeric | src=S&W [match:exact_norm]
- lasagna (norm='lasagna') <- FAO: 'Lasagna' | 1.042 g/cm^3 | raw='1.042' | kind=numeric | src=S&W [match:exact_norm]

### AMBIGUOUS (multiple candidates - need explicit resolution)

### UNMATCHED (no FAO entry found - NO guessed density)
- apple_pie (norm='apple pie')
- baby_back_ribs (norm='baby back ribs')
- baklava (norm='baklava')
- beef_carpaccio (norm='beef carpaccio')
- beef_tartare (norm='beef tartare')
- beet_salad (norm='beet salad')
- beignets (norm='beignets')
- bibimbap (norm='bibimbap')
- bread_pudding (norm='bread pudding')
- breakfast_burrito (norm='breakfast burrito')
- bruschetta (norm='bruschetta')
- caesar_salad (norm='caesar salad')
- cannoli (norm='cannoli')
- caprese_salad (norm='caprese salad')
- carrot_cake (norm='carrot cake')
- ceviche (norm='ceviche')
- cheese_plate (norm='cheese plate')
- cheesecake (norm='cheesecake')
- chicken_curry (norm='chicken curry')
- chicken_quesadilla (norm='chicken quesadilla')
- chicken_wings (norm='chicken wings')
- chocolate_cake (norm='chocolate cake')
- chocolate_mousse (norm='chocolate mousse')
- churros (norm='churros')
- clam_chowder (norm='clam chowder')
- club_sandwich (norm='club sandwich')
- crab_cakes (norm='crab cakes')
- creme_brulee (norm='creme brulee')
- croque_madame (norm='croque madame')
- cup_cakes (norm='cup cakes')
- deviled_eggs (norm='deviled eggs')
- donuts (norm='donuts')
- dumplings (norm='dumplings')
- edamame (norm='edamame')
- eggs_benedict (norm='eggs benedict')
- escargots (norm='escargots')
- falafel (norm='falafel')
- filet_mignon (norm='filet mignon')
- fish_and_chips (norm='fish and chips')
- foie_gras (norm='foie gras')
- french_fries (norm='french fries')
- french_onion_soup (norm='french onion soup')
- french_toast (norm='french toast')
- fried_calamari (norm='fried calamari')
- fried_rice (norm='fried rice')
- frozen_yogurt (norm='frozen yogurt')
- garlic_bread (norm='garlic bread')
- gnocchi (norm='gnocchi')
- greek_salad (norm='greek salad')
- grilled_cheese_sandwich (norm='grilled cheese sandwich')
- grilled_salmon (norm='grilled salmon')
- guacamole (norm='guacamole')
- gyoza (norm='gyoza')
- hamburger (norm='hamburger')
- hot_and_sour_soup (norm='hot and sour soup')
- hot_dog (norm='hot dog')
- huevos_rancheros (norm='huevos rancheros')
- hummus (norm='hummus')
- lobster_bisque (norm='lobster bisque')
- lobster_roll_sandwich (norm='lobster roll sandwich')
- macaroni_and_cheese (norm='macaroni and cheese')
- macarons (norm='macarons')
- miso_soup (norm='miso soup')
- mussels (norm='mussels')
- nachos (norm='nachos')
- omelette (norm='omelette')
- onion_rings (norm='onion rings')
- oysters (norm='oysters')
- pad_thai (norm='pad thai')
- paella (norm='paella')
- pancakes (norm='pancakes')
- panna_cotta (norm='panna cotta')
- peking_duck (norm='peking duck')
- pho (norm='pho')
- pizza (norm='pizza')
- pork_chop (norm='pork chop')
- poutine (norm='poutine')
- prime_rib (norm='prime rib')
- pulled_pork_sandwich (norm='pulled pork sandwich')
- ramen (norm='ramen')
- ravioli (norm='ravioli')
- red_velvet_cake (norm='red velvet cake')
- risotto (norm='risotto')
- samosa (norm='samosa')
- sashimi (norm='sashimi')
- scallops (norm='scallops')
- seaweed_salad (norm='seaweed salad')
- shrimp_and_grits (norm='shrimp and grits')
- spaghetti_bolognese (norm='spaghetti bolognese')
- spaghetti_carbonara (norm='spaghetti carbonara')
- spring_rolls (norm='spring rolls')
- steak (norm='steak')
- strawberry_shortcake (norm='strawberry shortcake')
- sushi (norm='sushi')
- tacos (norm='tacos')
- takoyaki (norm='takoyaki')
- tiramisu (norm='tiramisu')
- tuna_tartare (norm='tuna tartare')
- waffles (norm='waffles')
