# EasyFlux2KleinCondition Bucket List

This file records the fixed `1MP` ratio buckets used by `EasyFlux2KleinCondition`.

## Notes

- all bucket sizes below are based on `1MP`
- all width / height values are divisible by `16`
- `default` is not a fixed bucket
- non-`1MP` sizes are derived by scaling these buckets with `sqrt(target_mp)` and then aligning to `16`

## Fixed 1MP Buckets

| Ratio | Width | Height |
|------|------:|-------:|
| `1:1` | 1024 | 1024 |
| `4:3` | 1152 | 864 |
| `3:4` | 864 | 1152 |
| `3:2` | 1248 | 832 |
| `2:3` | 832 | 1248 |
| `5:4` | 1120 | 896 |
| `4:5` | 896 | 1120 |
| `16:9` | 1344 | 768 |
| `9:16` | 768 | 1344 |
| `2:1` | 1440 | 720 |
| `1:2` | 720 | 1440 |
| `21:9` | 1568 | 672 |
| `9:21` | 672 | 1568 |
| `4:1` | 2048 | 512 |
| `1:4` | 512 | 2048 |

## Special Case

`default` is not listed in the fixed table because it is resolved dynamically:

- with `img_01`, it follows `img_01` ratio rules
- without `img_01`, it falls back to `1:1`
