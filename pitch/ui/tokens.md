# AirTight UI tokens

Defined at the top of `pitch/ui/app.css`. Read from JS with
`getComputedStyle(document.documentElement).getPropertyValue('--name').trim()`; re-read when the theme changes.
The same semantic colour means the same thing in the 3D scene, the charts, the replay canvas and the text.

## Base values (light)
| token | value | meaning |
|---|---|---|
| `--base-paper` | `#f3f4f0` | survey-sheet ground, a cool neutral, not cream |
| `--base-ink` | `#14202b` | blue-black ink for text and rules |
| `--base-blue` | `#1f66b8` | the fleet family anchor |
| `--base-orange` | `#b84f08` | the intruder, the only warm colour |
| `--base-red` | `#c8102e` | alarm states only |
| `--base-grey` | `#8b8f94` | benign traffic |

## Semantic colours
| token | light | dark | meaning |
|---|---|---|---|
| `--ground` | `#f3f4f0` | `#12181e` | page and map ground |
| `--surface` | `#ffffff` | `#1a222a` | panels, chart and canvas background |
| `--surface-2` | `#e8ebe5` | `#232d37` | hover, wells, swarm message |
| `--hair` | `#cfd4cc` | `#36434f` | 1 px borders and dividers |
| `--grid` | `#dde1da` | `#26313b` | map and chart grid lines |
| `--fence` | `#3c4650` | `#aab4bd` | perimeter, gates, docks outlines |
| `--asset` | `#14202b` | `#eef1ee` | the protected asset (ink, identified by shape and label) |
| `--ink` | `#14202b` | `#eef1ee` | primary text, headline numbers, selected state |
| `--ink-2` | `#44515d` | `#b7c0c8` | secondary text, units, captions |
| `--ink-3` | `#5d6873` | `#8e9aa5` | tertiary text, axis labels, inactive tabs |
| `--fleet-drone` | `#1f66b8` | `#5598e6` | drones; also "deployed" and fleet trails |
| `--fleet-go2` | `#0a7273` | `#35c4b5` | Unitree Go2 |
| `--fleet-guard` | `#2a2a78` | `#c3bcff` | human guard |
| `--fleet-camera` | `#6790b5` | `#7d98b3` | fixed cameras and their wedges (graphics only, not text) |
| `--intruder` | `#b84f08` | `#f08a3c` | intruder, attack paths, red-team results |
| `--benign` | `#8b8f94` | `#6f767d` | benign traffic (graphics only) |
| `--alarm` | `#c8102e` | `#ff5a6a` | alarm fired / alert states. Nothing else is red |
| `--focus` | `#0b57d0` | `#8ab4ff` | keyboard focus ring |
| `--ramp-0..4` | `#dbe7f3 #a9c6e3 #6f9fcf #3b78b4 #174a86` | `#22364a #2f5479 #3f78ad #6fa3d6 #b3d2f0` | sequential ramp for priority or time; 0 is low/early and nearest the surface in both themes |

## Type, space, shape
| token | value |
|---|---|
| `--font` | Barlow (400 text, 500 labels, 600 strong) |
| `--display` | Barlow Semi Condensed 600, headings and figures |
| `--text-xs..3xl` | 12, 13, 15, 18, 24, 44, 72 px |
| `--w-text`, `--w-label`, `--w-strong` | 400, 500, 600 |
| `--space-1..8` | 4, 8, 12, 16, 24, 32, 48, 64 px |
| `--radius-1`, `--radius-2` | 2 px, 4 px |
| `--border` | `1px solid var(--hair)` |

Numbers: `font-variant-numeric: tabular-nums lining-nums` is set on `body`; canvas text must set its own font string.

## Legacy aliases (kept so existing scripts keep working; do not use in new code)
`--ink2`=`--ink-2`, `--ink3`=`--ink-3`, `--hair2`=`--hair`, `--ground2`=`--ground`, `--brand`/`--sec`/`--good`=`--fleet-drone`,
`--brand2`=`--fleet-go2`, `--atk`=`--intruder`, `--warn`=`--ramp-2`, `--on-brand`=`--surface`, `--r`/`--r2`=`--radius-2`, `--r3`=`--radius-1`.
`--shadow`, `--shadow2`, `--*-soft` no longer exist.

## Checks
`ui-review/tools/b/contrast.py` computes WCAG 2 contrast and simulates protan, deutan and tritan vision
(Machado 2009 matrices, severity 1) and prints the closest semantic pairs in CIELAB.
Text pairs: every ink and every text-capable semantic colour is at least 4.5:1 on `--surface` and `--ground` in both themes
(lowest: light `--intruder` 4.57 on ground, dark `--fleet-drone` about 5 on surface). `--fleet-camera`, `--benign`, `--hair` and `--grid` are graphics only (3:1 or decorative).
Colour-blind: fleet vs intruder vs alarm vs benign stay at dE76 of 14 or more under all three simulations; the closest pair is intruder/alarm under deutan (14),
so an alarm is never signalled by colour alone (it always carries the word or a ring marker). Inside the fleet family the types also differ in lightness and must differ in shape or label.
