# Workload Inspector Agent — Image Shape & Data Characteristic Analyzer

## Trigger
- Once at the start of an optimization campaign (persistent output)
- When adding new resolutions to the test set
- When considering shape-dependent optimizations

## Measurement Tasks

### 1. Shape Distribution
For all test resolutions (320x240 to 3840x2160):
- Width: min, p10, p50, p90, max
- Height: min, p10, p50, p90, max
- Total pixels: min, p10, p50, p90, max
- Aspect ratio: min, p10, p50, p90, max

Mark degenerate cases:
- Width or height < 32 (below patch_size * merge_size alignment)
- Extreme aspect ratios (>4:1 or <1:4)

### 2. Downscale Factor Distribution
- src->dst (448x448) scale factors
- Ranges: min, p50, max for width scale and height scale
- Identifies: which resolutions stress the bilinear cache most (large downscale = sparse src access)

### 3. Memory Footprint Per Resolution
| Resolution | Src Bytes | Norm Bytes | Patch Bytes | Total GPU |
|------------|-----------|------------|-------------|-----------|
| 320x240 | 0.2 MB | 2.4 MB | 4.8 MB | 7.4 MB |
| ... | ... | ... | ... | ... |
| 3840x2160 | 24.9 MB | 2.4 MB | 4.8 MB | 32.1 MB |

### 4. Source Stride Alignment
- For each resolution: is width * 3 aligned to 4 bytes? 16 bytes? 64 bytes?
- Misalignment affects coalescing efficiency and vectorization potential

## Output
Write to `experiments/workload_profile.md` (overwrite on each call):
```markdown
# Workload Profile — <Date>

## Top 5 Actionable Facts
1. <Fact — implication for kernel design>
2. ...
...

## Shape Distribution
| Metric | Min | P10 | P50 | P90 | Max |
|--------|-----|-----|-----|-----|-----|
| Width | | | | | |
| Height | | | | | |
| Pixels | | | | | |
| Aspect Ratio | | | | | |
| Downscale Factor | | | | | |

## Memory Footprint
<Table>

## Stride Alignment
<Per-resolution alignment analysis>

## Priority Optimization Hints
1. <Hint 1 — based on data characteristics>
2. <Hint 2>
```
